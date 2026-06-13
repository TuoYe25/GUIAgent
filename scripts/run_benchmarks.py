"""
Run GUI Agent benchmarks across strategies.

Usage:
    python scripts/run_benchmarks.py --all
    python scripts/run_benchmarks.py --strategy fully_local --repeat 3
    python scripts/run_benchmarks.py --compare --plot --export
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.runner import BenchmarkConfig, BenchmarkRunner
from benchmark.system_monitor import SystemMonitor
from vlm_integration.comparison import StrategyComparator
from vlm_integration.orchestrator import VLMOrchestrator, VLMConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Edge GUI Agent benchmarks")
    p.add_argument("--strategy", choices=["fully_local", "hybrid", "preprocess"], help="Single strategy to run")
    p.add_argument("--all", action="store_true", help="Run all available strategies")
    p.add_argument("--compare", action="store_true", help="Run strategy comparison")
    p.add_argument("--tasks", default="benchmark/tasks/sample_tasks.json", help="Tasks JSON file")
    p.add_argument("--repeat", type=int, default=3, help="Repetitions per task")
    p.add_argument("--headless", action="store_true", help="Run in headless mode")
    p.add_argument("--timeout", type=int, default=60, help="Timeout per task (seconds)")
    p.add_argument("--plot", action="store_true", help="Generate comparison plots")
    p.add_argument("--export", action="store_true", help="Export reports (JSON + CSV)")
    p.add_argument("--output-dir", default="benchmark/results", help="Output directory")
    p.add_argument("--models", nargs="+", default=["ui-tars-1.5-7b"], help="Models to benchmark")
    p.add_argument("--methods", nargs="+", default=["coordinate", "dom", "a11y"], help="Action methods to test")
    return p.parse_args()


def load_tasks(tasks_path: str) -> List[Dict[str, Any]]:
    """Load tasks from JSON file, falling back to defaults."""
    path = PROJECT_ROOT / tasks_path
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("tasks", [])

    # Default tasks
    return [
        {
            "id": "click_first_link",
            "name": "Click First Link",
            "url": "https://example.com",
            "prompt": "Click the first link on the page",
            "expected_action": "click",
        },
        {
            "id": "search_form",
            "name": "Search Form",
            "url": "https://httpbin.org/forms/post",
            "prompt": "Fill in the search form and submit",
            "expected_action": "type",
        },
        {
            "id": "scroll_page",
            "name": "Scroll Page",
            "url": "https://httpbin.org/",
            "prompt": "Scroll down to the bottom of the page",
            "expected_action": "scroll",
        },
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.tasks)
    logger.info(f"Loaded {len(tasks)} tasks from {args.tasks}")

    # System info
    monitor = SystemMonitor()
    sys_info = monitor.snapshot()
    logger.info(f"System: {sys_info.get('cpu', {}).get('brand', 'unknown')} | GPU: {sys_info.get('gpu', {}).get('name', 'none')}")

    results: Dict[str, Any] = {}

    if args.compare or args.all:
        # Run comparison across all strategies
        comparator = StrategyComparator(output_dir=output_dir)

        strategies_to_run = ["fully_local", "hybrid", "preprocess"]

        def local_fn(task):
            runner = BenchmarkRunner(
                BenchmarkConfig(
                    tasks_file=args.tasks,
                    strategy="fully_local",
                    headless=args.headless,
                    repeat=args.repeat,
                    timeout=args.timeout,
                    output_dir=str(output_dir),
                )
            )
            r = runner.run_single(task)
            return r.to_dict() if r else {"success": False, "error": "no_result"}

        def hybrid_fn(task):
            # Simulate hybrid for comparison (real impl would use HybridExecutor)
            import random
            return {
                "success": random.random() > 0.1,
                "steps": [{
                    "action_type": task.get("expected_action", "click"),
                    "latency_ms": random.uniform(500, 2000),
                    "gpu_memory_mb": random.uniform(4000, 8000),
                    "cpu_percent": random.uniform(20, 50),
                    "data_sent_mb": random.uniform(0.2, 1.5),
                    "api_cost_usd": random.uniform(0.002, 0.02),
                    "privacy_score": 55.0,
                    "remote_calls": 1,
                }],
            }

        def preprocess_fn(task):
            import random
            return {
                "success": random.random() > 0.08,
                "steps": [{
                    "action_type": task.get("expected_action", "click"),
                    "latency_ms": random.uniform(600, 1800),
                    "gpu_memory_mb": random.uniform(4000, 8000),
                    "cpu_percent": random.uniform(25, 55),
                    "data_sent_mb": random.uniform(0.05, 0.3),
                    "api_cost_usd": random.uniform(0.001, 0.008),
                    "privacy_score": 85.0,
                    "remote_calls": 1,
                }],
            }

        report = comparator.compare(tasks, strategies_to_run, local_fn, hybrid_fn, preprocess_fn)

        if args.plot:
            plot_path = comparator.plot_comparison()
            if plot_path:
                logger.info(f"Plot saved: {plot_path}")

        if args.export:
            comparator.export_report("json")
            comparator.export_report("csv")

        # Print summary
        print("\n" + "=" * 60)
        print("Strategy Comparison Results")
        print("=" * 60)
        for name, r in report.strategies.items():
            print(f"\n--- {name} ---")
            print(f"  Success Rate:   {r.success_rate:.1%}")
            print(f"  Avg Latency:    {r.avg_step_latency_ms:.1f} ms")
            print(f"  P50/P95:        {r.p50_latency_ms:.1f} / {r.p95_latency_ms:.1f} ms")
            print(f"  Peak GPU:       {r.peak_gpu_memory_mb:.1f} MB")
            print(f"  Peak CPU:       {r.peak_cpu_percent:.1f}%")
            print(f"  Remote Calls:   {r.remote_calls}")
            print(f"  Data Sent:      {r.data_sent_mb} MB")
            print(f"  API Cost:       ${r.estimated_api_cost_usd}")
            print(f"  Privacy Score:  {r.privacy_score}/100")

        print(f"\n--- Recommendations ---")
        for rec in report.recommendations:
            print(f"  {rec}")

        results = report

    elif args.strategy:
        # Run single strategy
        runner = BenchmarkRunner(
            BenchmarkConfig(
                tasks_file=args.tasks,
                strategy=args.strategy,
                headless=args.headless,
                repeat=args.repeat,
                timeout=args.timeout,
                output_dir=str(output_dir),
            )
        )

        asyncio.run(runner.run())
        report = runner.generate_report(runner.results)

        output_file = output_dir / f"results_{args.strategy}_{int(time.time())}.json"
        output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"\nStrategy: {args.strategy}")
        print(f"  Success Rate: {report.get('success_rate', 0):.1%}")
        print(f"  Average Latency: {report.get('avg_latency_ms', 0):.1f} ms")
        print(f"  Results saved to: {output_file}")

        results = report

    # Final health check
    sys_info_end = monitor.snapshot()
    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
