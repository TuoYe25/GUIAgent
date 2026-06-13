"""
Comparison analysis for benchmark results.

Compare:
- Different models (UI-TARS vs ShowUI vs OS-ATLAS)
- Different action methods (coordinate vs DOM vs a11y)
- Traditional competitors (Playwright vs Selenium vs Puppeteer)
- Hybrid approaches (local GUI + remote VLM)
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger


# ---------------------------------------------------------------------------
# Comparison Types
# ---------------------------------------------------------------------------

@dataclass
class ComparisonGroup:
    """A group of benchmark results to compare."""

    name: str
    results: List[Dict[str, Any]]
    color: str = "blue"
    marker: str = "o"

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics for this group."""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        summary: Dict[str, Any] = {
            "count": len(self.results),
            "success_rate": df["success"].mean() if "success" in df.columns else None,
            "avg_steps": df["steps"].mean() if "steps" in df.columns else None,
            "avg_time_ms": df["execution_time_ms"].mean() if "execution_time_ms" in df.columns else None,
            "avg_cpu": df["cpu_usage_percent"].mean() if "cpu_usage_percent" in df.columns else None,
            "avg_gpu_mb": df["gpu_memory_mib"].mean() if "gpu_memory_mib" in df.columns else None,
            "avg_mem_mb": df["system_memory_mib"].mean() if "system_memory_mib" in df.columns else None,
        }
        return summary


@dataclass
class ComparisonResult:
    """Result of comparing multiple groups."""

    groups: Dict[str, ComparisonGroup]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "groups": {name: group.summary() for name, group in self.groups.items()},
            "metrics": self.metrics,
        }

    def save(self, path: Path) -> None:
        """Save comparison result to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved comparison: {path}")


# ---------------------------------------------------------------------------
# Comparison Engine
# ---------------------------------------------------------------------------

class ComparisonEngine:
    """Compare benchmark results across different dimensions."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.results: List[Dict[str, Any]] = []
        self._load_results()

    def _load_results(self) -> None:
        """Load all JSON results from the directory."""
        for json_file in self.results_dir.glob("*.json"):
            if "summary" in json_file.name:
                continue
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.results.extend(data)
                    else:
                        self.results.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(self.results)} benchmark results")

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def group_by_model(self) -> Dict[str, ComparisonGroup]:
        """Group results by model name."""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for result in self.results:
            model = result.get("model_name", "unknown")
            groups.setdefault(model, []).append(result)

        return {
            model: ComparisonGroup(name=model, results=results)
            for model, results in groups.items()
        }

    def group_by_method(self) -> Dict[str, ComparisonGroup]:
        """Group results by action method."""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for result in self.results:
            method = result.get("action_method", "unknown")
            groups.setdefault(method, []).append(result)

        return {
            method: ComparisonGroup(name=method, results=results)
            for method, results in groups.items()
        }

    def group_by_task(self) -> Dict[str, ComparisonGroup]:
        """Group results by task ID."""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for result in self.results:
            task = result.get("task_id", "unknown")
            groups.setdefault(task, []).append(result)

        return {
            task: ComparisonGroup(name=task, results=results)
            for task, results in groups.items()
        }

    def group_by_difficulty(self, task_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, ComparisonGroup]:
        """Group results by task difficulty."""
        # Default difficulty mapping
        difficulty_map = {
            "click_button_basic": "easy",
            "type_text_basic": "easy",
            "scroll_down_basic": "easy",
            "navigate_to_url": "easy",
            "click_link_navigation": "easy",
            "click_checkbox": "easy",
            "type_password": "easy",
            "navigate_back": "easy",
            "navigate_refresh": "easy",
            "click_small_element": "medium",
            "click_dropdown_option": "medium",
            "type_multiline": "medium",
            "type_search": "medium",
            "scroll_to_element": "medium",
            "form_login": "medium",
            "search_and_click_result": "medium",
            "complex_tabular_data": "medium",
            "complex_modal_dialog": "medium",
            "form_registration": "hard",
            "form_with_validation": "hard",
            "search_filtered": "hard",
            "complex_shopping_cart": "hard",
            "complex_book_flight": "extreme",
        }

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for result in self.results:
            task_id = result.get("task_id", "")
            difficulty = difficulty_map.get(task_id, "unknown")
            groups.setdefault(difficulty, []).append(result)

        return {
            diff: ComparisonGroup(name=diff, results=results)
            for diff, results in groups.items()
        }

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_models(self) -> ComparisonResult:
        """Compare performance across different models."""
        groups = self.group_by_model()
        metrics = self._compute_comparison_metrics(groups)
        return ComparisonResult(groups=groups, metrics=metrics)

    def compare_methods(self) -> ComparisonResult:
        """Compare performance across action methods."""
        groups = self.group_by_method()
        metrics = self._compute_comparison_metrics(groups)
        return ComparisonResult(groups=groups, metrics=metrics)

    def compare_tasks(self) -> ComparisonResult:
        """Compare performance across tasks."""
        groups = self.group_by_task()
        metrics = self._compute_comparison_metrics(groups)
        return ComparisonResult(groups=groups, metrics=metrics)

    def compare_difficulty(self) -> ComparisonResult:
        """Compare performance by task difficulty."""
        groups = self.group_by_difficulty()
        metrics = self._compute_comparison_metrics(groups)
        return ComparisonResult(groups=groups, metrics=metrics)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_comparison_metrics(self, groups: Dict[str, ComparisonGroup]) -> Dict[str, Any]:
        """Compute comparison metrics between groups."""
        metrics: Dict[str, Any] = {}

        # Success rate comparison
        success_rates = {}
        for name, group in groups.items():
            summary = group.summary()
            if summary.get("success_rate") is not None:
                success_rates[name] = summary["success_rate"]

        if success_rates:
            metrics["success_rate"] = {
                "values": success_rates,
                "best": max(success_rates, key=success_rates.get) if success_rates else None,
                "worst": min(success_rates, key=success_rates.get) if success_rates else None,
                "range": max(success_rates.values()) - min(success_rates.values()) if success_rates else 0,
            }

        # Latency comparison
        latencies = {}
        for name, group in groups.items():
            summary = group.summary()
            if summary.get("avg_time_ms") is not None:
                latencies[name] = summary["avg_time_ms"]

        if latencies:
            metrics["latency_ms"] = {
                "values": latencies,
                "fastest": min(latencies, key=latencies.get) if latencies else None,
                "slowest": max(latencies, key=latencies.get) if latencies else None,
                "range": max(latencies.values()) - min(latencies.values()) if latencies else 0,
            }

        # Resource usage comparison
        cpu_usage = {}
        gpu_usage = {}
        mem_usage = {}
        for name, group in groups.items():
            summary = group.summary()
            if summary.get("avg_cpu") is not None:
                cpu_usage[name] = summary["avg_cpu"]
            if summary.get("avg_gpu_mb") is not None:
                gpu_usage[name] = summary["avg_gpu_mb"]
            if summary.get("avg_mem_mb") is not None:
                mem_usage[name] = summary["avg_mem_mb"]

        if cpu_usage:
            metrics["cpu_usage_percent"] = cpu_usage
        if gpu_usage:
            metrics["gpu_memory_mb"] = gpu_usage
        if mem_usage:
            metrics["system_memory_mb"] = mem_usage

        return metrics

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_success_rate(self, groups: Dict[str, ComparisonGroup], save_path: Optional[Path] = None) -> None:
        """Plot success rate comparison."""
        names = []
        rates = []
        colors = []

        for name, group in groups.items():
            summary = group.summary()
            rate = summary.get("success_rate")
            if rate is not None:
                names.append(name)
                rates.append(rate * 100)  # to percentage
                colors.append(group.color)

        if not rates:
            logger.warning("No success rate data to plot")
            return

        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, rates, color=colors, edgecolor="black")
        plt.xlabel("Group")
        plt.ylabel("Success Rate (%)")
        plt.title("Success Rate Comparison")
        plt.ylim(0, 105)
        plt.xticks(rotation=45)

        # Add value labels
        for bar, rate in zip(bars, rates):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved success rate plot: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_latency_comparison(self, groups: Dict[str, ComparisonGroup], save_path: Optional[Path] = None) -> None:
        """Plot latency comparison."""
        names = []
        latencies = []
        colors = []

        for name, group in groups.items():
            summary = group.summary()
            latency = summary.get("avg_time_ms")
            if latency is not None:
                names.append(name)
                latencies.append(latency)
                colors.append(group.color)

        if not latencies:
            logger.warning("No latency data to plot")
            return

        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, latencies, color=colors, edgecolor="black")
        plt.xlabel("Group")
        plt.ylabel("Average Latency (ms)")
        plt.title("Latency Comparison")
        plt.xticks(rotation=45)

        # Add value labels
        for bar, latency in zip(bars, latencies):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{latency:.0f} ms",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved latency plot: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_resource_comparison(self, groups: Dict[str, ComparisonGroup], save_path: Optional[Path] = None) -> None:
        """Plot resource usage comparison (CPU, GPU, Memory)."""
        names = list(groups.keys())
        cpu_vals = []
        gpu_vals = []
        mem_vals = []

        for name, group in groups.items():
            summary = group.summary()
            cpu_vals.append(summary.get("avg_cpu", 0))
            gpu_vals.append(summary.get("avg_gpu_mb", 0))
            mem_vals.append(summary.get("avg_mem_mb", 0))

        x = range(len(names))
        width = 0.25

        plt.figure(figsize=(14, 7))
        plt.bar([i - width for i in x], cpu_vals, width, label="CPU %", color="skyblue")
        plt.bar(x, gpu_vals, width, label="GPU MB", color="lightcoral")
        plt.bar([i + width for i in x], mem_vals, width, label="Memory MB", color="lightgreen")

        plt.xlabel("Group")
        plt.ylabel("Resource Usage")
        plt.title("Resource Usage Comparison")
        plt.xticks(x, names, rotation=45)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved resource plot: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_scatter_efficiency(self, groups: Dict[str, ComparisonGroup], save_path: Optional[Path] = None) -> None:
        """Scatter plot: success rate vs latency."""
        names = []
        success_rates = []
        latencies = []
        colors = []

        for name, group in groups.items():
            summary = group.summary()
            rate = summary.get("success_rate")
            latency = summary.get("avg_time_ms")
            if rate is not None and latency is not None:
                names.append(name)
                success_rates.append(rate * 100)
                latencies.append(latency)
                colors.append(group.color)

        if not success_rates:
            logger.warning("No data for scatter plot")
            return

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latencies, success_rates, c=colors, s=100, alpha=0.7, edgecolors="black")

        # Add labels
        for i, name in enumerate(names):
            plt.annotate(
                name,
                (latencies[i], success_rates[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        plt.xlabel("Average Latency (ms)")
        plt.ylabel("Success Rate (%)")
        plt.title("Efficiency: Success Rate vs Latency")
        plt.grid(True, alpha=0.3)

        # Ideal quadrant
        plt.axvline(x=statistics.median(latencies) if latencies else 0, color="gray", linestyle="--", alpha=0.5)
        plt.axhline(y=statistics.median(success_rates) if success_rates else 0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved scatter plot: {save_path}")
        else:
            plt.show()
        plt.close()

    # ------------------------------------------------------------------
    # Report Generation
    # ------------------------------------------------------------------

    def generate_report(self, output_dir: Path) -> None:
        """Generate a comprehensive comparison report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run comparisons
        model_comparison = self.compare_models()
        method_comparison = self.compare_methods()
        task_comparison = self.compare_tasks()
        difficulty_comparison = self.compare_difficulty()

        # Save JSON
        comparisons = {
            "models": model_comparison.to_dict(),
            "methods": method_comparison.to_dict(),
            "tasks": task_comparison.to_dict(),
            "difficulty": difficulty_comparison.to_dict(),
        }

        report_path = output_dir / "comparison_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(comparisons, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved comparison report: {report_path}")

        # Generate plots
        self.plot_success_rate(model_comparison.groups, output_dir / "success_rate_models.png")
        self.plot_latency_comparison(model_comparison.groups, output_dir / "latency_models.png")
        self.plot_resource_comparison(model_comparison.groups, output_dir / "resources_models.png")
        self.plot_scatter_efficiency(model_comparison.groups, output_dir / "efficiency_models.png")

        self.plot_success_rate(method_comparison.groups, output_dir / "success_rate_methods.png")
        self.plot_latency_comparison(method_comparison.groups, output_dir / "latency_methods.png")

        # Generate markdown summary
        self._generate_markdown_summary(comparisons, output_dir / "summary.md")

    def _generate_markdown_summary(self, comparisons: Dict[str, Any], output_path: Path) -> None:
        """Generate a markdown summary of comparisons."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# GUI Agent Benchmark Comparison Report\n\n")

            # Model comparison
            f.write("## Model Comparison\n\n")
            model_data = comparisons.get("models", {})
            for model, stats in model_data.get("groups", {}).items():
                f.write(f"### {model}\n")
                f.write(f"- Success rate: {stats.get('success_rate', 0):.1%}\n")
                f.write(f"- Average latency: {stats.get('avg_time_ms', 0):.0f} ms\n")
                f.write(f"- Average CPU usage: {stats.get('avg_cpu', 0):.1f}%\n")
                f.write(f"- Average GPU memory: {stats.get('avg_gpu_mb', 0):.0f} MB\n\n")

            # Method comparison
            f.write("## Action Method Comparison\n\n")
            method_data = comparisons.get("methods", {})
            for method, stats in method_data.get("groups", {}).items():
                f.write(f"### {method}\n")
                f.write(f"- Success rate: {stats.get('success_rate', 0):.1%}\n")
                f.write(f"- Average latency: {stats.get('avg_time_ms', 0):.0f} ms\n\n")

            # Key findings
            f.write("## Key Findings\n\n")
            f.write("1. **Best performing model**: TODO\n")
            f.write("2. **Most efficient method**: TODO\n")
            f.write("3. **Hardest task**: TODO\n")
            f.write("4. **Resource usage**: TODO\n")

        logger.info(f"Saved markdown summary: {output_path}")
