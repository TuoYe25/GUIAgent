"""
VLM Strategy Comparison — Benchmark local vs hybrid approaches.

Compares three strategies:
1. FULLY_LOCAL  — Edge VLM planner + local GUI model
2. HYBRID       — Local GUI model + remote VLM/LLM planner-evaluator
3. PREPROCESS   — Preprocess data locally + send structured data to remote

Metrics:
- End-to-end task success rate
- Average latency per step
- GPU/CPU/memory usage
- Data sent to remote (privacy exposure)
- API cost estimation
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    """Single strategy benchmark result."""
    strategy: str
    tasks_completed: int = 0
    tasks_total: int = 0
    success_rate: float = 0.0

    # Latency (ms)
    avg_step_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Resource (peak)
    peak_gpu_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0

    # Remote usage
    remote_calls: int = 0
    data_sent_mb: float = 0.0
    estimated_api_cost_usd: float = 0.0

    # Privacy score (0-100, higher = better privacy)
    privacy_score: float = 100.0

    # Per-task breakdown
    task_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Full comparison report across strategies."""
    strategies: Dict[str, StrategyResult] = field(default_factory=dict)
    tasks: List[str] = field(default_factory=list)
    best_latency: Optional[str] = None
    best_success_rate: Optional[str] = None
    best_privacy: Optional[str] = None
    best_cost_efficiency: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# API Cost Estimator
# ---------------------------------------------------------------------------

# Approximate pricing per 1M tokens (USD), as of mid-2024
API_PRICING: Dict[str, Dict[str, float]] = {
    "openai": {
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    },
    "anthropic": {
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    },
    "groq": {
        "llama3-70b-8192": {"input": 0.59, "output": 0.79},
        "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    },
}


def estimate_cost(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> float:
    """Estimate API cost for a call."""
    pricing = API_PRICING.get(provider, {}).get(model)
    if not pricing:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Privacy Score Calculator
# ---------------------------------------------------------------------------

def calculate_privacy_score(
    local_only: bool,
    data_sent_mb: float,
    redactions: int,
    uses_remote: bool,
) -> float:
    """
    Calculate a privacy score from 0-100.

    - 100: Fully local, no data leaves device
    - 0: All raw data sent to remote without any preprocessing
    """
    score = 100.0

    if uses_remote:
        # Penalty for using remote
        score -= 30

        # Penalty proportional to data sent
        score -= min(30, data_sent_mb * 10)

        # Bonus for redactions / preprocessing
        score += min(30, redactions * 2)

    return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# Strategy Comparator
# ---------------------------------------------------------------------------

class StrategyComparator:
    """
    Compare local, hybrid, and preprocessor strategies end-to-end.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path("benchmark/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report: Optional[ComparisonReport] = None

    # ------------------------------------------------------------------
    # Run Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        tasks: List[Dict[str, Any]],
        strategies: List[str],
        local_executor_fn: Callable,
        hybrid_executor_fn: Optional[Callable] = None,
        preprocessor_executor_fn: Optional[Callable] = None,
    ) -> ComparisonReport:
        """
        Run all strategies on the same tasks and compare results.

        Args:
            tasks: List of task definitions
            strategies: List of strategy names to run
            local_executor_fn: Function to run fully-local strategy
                signature: fn(task: Dict) -> Dict with 'success', 'latency_ms', 'resource' keys
            hybrid_executor_fn: Function for hybrid strategy
            preprocessor_executor_fn: Function for preprocessor strategy

        Returns:
            ComparisonReport
        """
        report = ComparisonReport()
        report.tasks = [t.get("name", t.get("description", f"task_{i}")) for i, t in enumerate(tasks)]

        strategy_fns = {
            "fully_local": local_executor_fn,
            "hybrid": hybrid_executor_fn or local_executor_fn,
            "preprocess": preprocessor_executor_fn or local_executor_fn,
        }

        for strategy in strategies:
            if strategy not in strategy_fns:
                continue

            logger.info(f"Running strategy: {strategy}")
            fn = strategy_fns[strategy]
            result = self._run_strategy(strategy, tasks, fn)
            report.strategies[strategy] = result

        # Determine bests
        self._determine_bests(report)
        self.report = report

        return report

    def _run_strategy(
        self, strategy: str, tasks: List[Dict], executor_fn: Callable
    ) -> StrategyResult:
        """Execute all tasks under one strategy."""
        result = StrategyResult(strategy=strategy, tasks_total=len(tasks))

        all_latencies: List[float] = []
        peak_gpu = 0.0
        peak_cpu = 0.0
        total_data_sent = 0.0
        total_cost = 0.0
        remote_calls = 0
        privacy_scores: List[float] = []

        for task in tasks:
            try:
                task_result = executor_fn(task)
            except Exception as e:
                logger.error(f"Task failed [{strategy}]: {e}")
                task_result = {"success": False, "error": str(e)}

            result.task_results.append(task_result)

            if task_result.get("success", False):
                result.tasks_completed += 1

            # Collect metrics
            if "steps" in task_result:
                for step in task_result["steps"]:
                    lat = step.get("latency_ms", step.get("total_latency_ms", 0))
                    if lat > 0:
                        all_latencies.append(lat)

                    peak_gpu = max(peak_gpu, step.get("gpu_memory_mb", step.get("local_gpu_memory_mb", 0)))
                    peak_cpu = max(peak_cpu, step.get("cpu_percent", step.get("local_cpu_percent", 0)))
                    total_data_sent += step.get("data_sent_mb", 0)
                    total_cost += step.get("api_cost_usd", 0)
                    remote_calls += step.get("remote_calls", 0)
                    privacy_scores.append(step.get("privacy_score", 100.0))

            elif "latency_ms" in task_result:
                all_latencies.append(task_result["latency_ms"])
                peak_gpu = max(peak_gpu, task_result.get("gpu_memory_mb", 0))
                peak_cpu = max(peak_cpu, task_result.get("cpu_percent", 0))

        # Aggregate
        result.success_rate = result.tasks_completed / max(result.tasks_total, 1)

        if all_latencies:
            arr = np.array(all_latencies)
            result.avg_step_latency_ms = round(float(np.mean(arr)), 1)
            result.p50_latency_ms = round(float(np.percentile(arr, 50)), 1)
            result.p95_latency_ms = round(float(np.percentile(arr, 95)), 1)
            result.p99_latency_ms = round(float(np.percentile(arr, 99)), 1)

        result.peak_gpu_memory_mb = round(peak_gpu, 1)
        result.peak_cpu_percent = round(peak_cpu, 1)
        result.remote_calls = remote_calls
        result.data_sent_mb = round(total_data_sent, 2)
        result.estimated_api_cost_usd = round(total_cost, 6)

        result.privacy_score = (
            round(float(np.mean(privacy_scores)), 1) if privacy_scores
            else 100.0 if strategy == "fully_local" else 50.0
        )

        return result

    def _determine_bests(self, report: ComparisonReport) -> None:
        """Determine which strategy wins in each category."""
        if not report.strategies:
            return

        best_latency = min(report.strategies.items(), key=lambda x: x[1].avg_step_latency_ms or float("inf"))
        best_success = max(report.strategies.items(), key=lambda x: x[1].success_rate or 0)
        best_privacy = max(report.strategies.items(), key=lambda x: x[1].privacy_score or 0)
        best_cost = min(report.strategies.items(), key=lambda x: x[1].estimated_api_cost_usd or float("inf"))

        report.best_latency = best_latency[0]
        report.best_success_rate = best_success[0]
        report.best_privacy = best_privacy[0]
        report.best_cost_efficiency = best_cost[0]

        # Generate recommendations
        report.recommendations = [
            f"Best latency: {best_latency[0]} ({best_latency[1].avg_step_latency_ms}ms avg)",
            f"Best success rate: {best_success[0]} ({best_success[1].success_rate:.1%})",
            f"Best privacy: {best_privacy[0]} (score: {best_privacy[1].privacy_score})",
            f"Best cost efficiency: {best_cost[0]} (${best_cost[1].estimated_api_cost_usd})",
        ]

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_comparison(self) -> Optional[Path]:
        """Generate comparison charts."""
        if not self.report or not self.report.strategies:
            return None

        if not HAS_MPL:
            logger.warning("matplotlib not available, skipping plot generation")
            return None

        strategies = list(self.report.strategies.keys())
        n = len(strategies)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("GUI Agent Strategy Comparison", fontsize=14, fontweight="bold")

        # 1. Latency comparison (bar)
        ax1 = axes[0, 0]
        latencies = [self.report.strategies[s].avg_step_latency_ms for s in strategies]
        bars1 = ax1.bar(strategies, latencies, color=["#4CAF50", "#2196F3", "#FF9800"][:n])
        ax1.set_title("Average Step Latency (ms)")
        ax1.set_ylabel("ms")
        for bar, val in zip(bars1, latencies):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{val:.0f}", ha="center", fontsize=9)

        # 2. Success rate (bar)
        ax2 = axes[0, 1]
        success_rates = [self.report.strategies[s].success_rate * 100 for s in strategies]
        bars2 = ax2.bar(strategies, success_rates, color=["#4CAF50", "#2196F3", "#FF9800"][:n])
        ax2.set_title("Success Rate (%)")
        ax2.set_ylabel("%")
        ax2.set_ylim(0, 105)
        for bar, val in zip(bars2, success_rates):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontsize=9)

        # 3. Resource usage
        ax3 = axes[1, 0]
        x = np.arange(n)
        width = 0.35
        gpu_vals = [self.report.strategies[s].peak_gpu_memory_mb for s in strategies]
        cpu_vals = [self.report.strategies[s].peak_cpu_percent for s in strategies]
        ax3.bar(x - width / 2, gpu_vals, width, label="GPU Memory (MB)", color="#E91E63")
        ax3.bar(x + width / 2, cpu_vals, width, label="CPU (%)", color="#9C27B0")
        ax3.set_title("Resource Usage (Peak)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies, fontsize=9)
        ax3.legend(fontsize=8)

        # 4. Privacy & cost
        ax4 = axes[1, 1]
        privacy_vals = [self.report.strategies[s].privacy_score for s in strategies]
        cost_vals = [self.report.strategies[s].estimated_api_cost_usd * 1000 for s in strategies]  # scale to milli
        ax4_bars1 = ax4.bar(x - width / 2, privacy_vals, width, label="Privacy Score", color="#00BCD4")
        ax4_bars2 = ax4.bar(x + width / 2, cost_vals, width, label="API Cost (milli-USD)", color="#FF5722")
        ax4.set_title("Privacy & Cost")
        ax4.set_xticks(x)
        ax4.set_xticklabels(strategies, fontsize=9)
        ax4.legend(fontsize=8)

        plt.tight_layout()
        path = self.output_dir / "strategy_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Comparison chart saved to {path}")
        return path

    # ------------------------------------------------------------------
    # Report Export
    # ------------------------------------------------------------------

    def export_report(self, format: str = "json") -> Optional[Path]:
        """Export comparison report."""
        if not self.report:
            return None

        if format == "json":
            data = self._report_to_dict(self.report)
            path = self.output_dir / "comparison_report.json"
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        elif format == "csv":
            path = self.output_dir / "comparison_report.csv"
            lines = [
                "strategy,success_rate,avg_latency_ms,p50_ms,p95_ms,peak_gpu_mb,peak_cpu_pct,remote_calls,data_sent_mb,api_cost_usd,privacy_score"
            ]
            for name, r in self.report.strategies.items():
                lines.append(
                    f"{name},{r.success_rate},{r.avg_step_latency_ms},{r.p50_latency_ms},"
                    f"{r.p95_latency_ms},{r.peak_gpu_memory_mb},{r.peak_cpu_percent},"
                    f"{r.remote_calls},{r.data_sent_mb},{r.estimated_api_cost_usd},{r.privacy_score}"
                )
            path.write_text("\n".join(lines), encoding="utf-8")

        else:
            return None

        logger.info(f"Report exported to {path}")
        return path

    def _report_to_dict(self, report: ComparisonReport) -> Dict[str, Any]:
        strategies = {}
        for name, r in report.strategies.items():
            strategies[name] = {
                "success_rate": r.success_rate,
                "avg_step_latency_ms": r.avg_step_latency_ms,
                "p50_latency_ms": r.p50_latency_ms,
                "p95_latency_ms": r.p95_latency_ms,
                "p99_latency_ms": r.p99_latency_ms,
                "peak_gpu_memory_mb": r.peak_gpu_memory_mb,
                "peak_cpu_percent": r.peak_cpu_percent,
                "remote_calls": r.remote_calls,
                "data_sent_mb": r.data_sent_mb,
                "estimated_api_cost_usd": r.estimated_api_cost_usd,
                "privacy_score": r.privacy_score,
            }

        return {
            "strategies": strategies,
            "tasks": report.tasks,
            "best_latency": report.best_latency,
            "best_success_rate": report.best_success_rate,
            "best_privacy": report.best_privacy,
            "best_cost_efficiency": report.best_cost_efficiency,
            "recommendations": report.recommendations,
        }


# ---------------------------------------------------------------------------
# Quick CLI
# ---------------------------------------------------------------------------

def run_quick_comparison() -> None:
    """Run a quick hardcoded comparison for testing."""
    comparator = StrategyComparator()

    # Dummy tasks
    tasks = [
        {"name": "Search GitHub for UI-TARS", "url": "https://github.com/search?q=ui-tars"},
        {"name": "Fill sample form", "url": "https://example.com/form"},
        {"name": "Navigate menu", "url": "https://example.com"},
    ]

    def local_fn(task):
        import random
        return {
            "success": random.random() > 0.1,
            "latency_ms": random.uniform(200, 800),
            "gpu_memory_mb": random.uniform(2000, 5000),
            "cpu_percent": random.uniform(20, 60),
            "privacy_score": 100.0,
        }

    def hybrid_fn(task):
        import random
        return {
            "success": random.random() > 0.05,
            "steps": [{
                "latency_ms": random.uniform(500, 2000),
                "gpu_memory_mb": random.uniform(2000, 4000),
                "cpu_percent": random.uniform(15, 50),
                "data_sent_mb": random.uniform(0.1, 2.0),
                "api_cost_usd": random.uniform(0.001, 0.05),
                "privacy_score": 60.0,
                "remote_calls": 1,
            }],
        }

    report = comparator.compare(tasks, ["fully_local", "hybrid"], local_fn, hybrid_fn)
    comparator.plot_comparison()
    comparator.export_report("json")
    comparator.export_report("csv")

    print(f"\n=== Comparison Complete ===")
    for rec in report.recommendations:
        print(f"  {rec}")


if __name__ == "__main__":
    run_quick_comparison()
