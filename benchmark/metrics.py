"""
Metrics collection and analysis for GUI Agent benchmarks.

Collects:
- Task success rate
- Step efficiency
- Latency (end-to-end, inference, parsing)
- System resources (CPU, GPU, memory)
- Token usage (for LLM/VLM models)
- Cost estimation (if applicable)
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Metric Types
# ---------------------------------------------------------------------------

class MetricType(str, Enum):
    """Categories of metrics."""
    SUCCESS = "success"
    LATENCY = "latency"
    RESOURCE = "resource"
    EFFICIENCY = "efficiency"
    COST = "cost"
    QUALITY = "quality"


@dataclass
class Metric:
    """A single metric measurement."""

    name: str
    value: float
    unit: str
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "unit": self.unit,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class MetricAggregate:
    """Aggregate statistics for a metric across runs."""

    name: str
    unit: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    metric_type: MetricType

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "count": self.count,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "median": round(self.median, 4),
            "q25": round(self.q25, 4),
            "q75": round(self.q75, 4),
            "type": self.metric_type.value,
        }


# ---------------------------------------------------------------------------
# Metrics Collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Collect and aggregate metrics from benchmark runs."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.metrics: Dict[str, List[Metric]] = {}
        self.output_dir = output_dir or Path("benchmark/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record(
        self,
        name: str,
        value: float,
        unit: str,
        metric_type: MetricType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a single metric."""
        metric = Metric(
            name=name,
            value=value,
            unit=unit,
            metric_type=metric_type,
            metadata=metadata or {},
        )
        self.metrics.setdefault(name, []).append(metric)
        logger.debug(f"Metric recorded: {name}={value} {unit}")

    def record_success(
        self,
        success: bool,
        task_id: str,
        model_name: str,
        action_method: str,
        steps: int,
    ) -> None:
        """Record task success/failure."""
        self.record(
            name="task_success",
            value=1.0 if success else 0.0,
            unit="bool",
            metric_type=MetricType.SUCCESS,
            metadata={
                "task_id": task_id,
                "model": model_name,
                "method": action_method,
                "steps": steps,
            },
        )

    def record_latency(
        self,
        latency_ms: float,
        stage: str,
        task_id: str,
        model_name: str,
    ) -> None:
        """Record latency for a specific stage."""
        self.record(
            name=f"latency_{stage}",
            value=latency_ms,
            unit="ms",
            metric_type=MetricType.LATENCY,
            metadata={"task_id": task_id, "model": model_name, "stage": stage},
        )

    def record_resource(
        self,
        cpu_percent: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None,
        sys_memory_mb: Optional[float] = None,
        gpu_util_percent: Optional[float] = None,
    ) -> None:
        """Record system resource usage."""
        if cpu_percent is not None:
            self.record("cpu_usage", cpu_percent, "%", MetricType.RESOURCE)
        if gpu_memory_mb is not None:
            self.record("gpu_memory", gpu_memory_mb, "MB", MetricType.RESOURCE)
        if sys_memory_mb is not None:
            self.record("system_memory", sys_memory_mb, "MB", MetricType.RESOURCE)
        if gpu_util_percent is not None:
            self.record("gpu_utilization", gpu_util_percent, "%", MetricType.RESOURCE)

    def record_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
        task_id: str,
    ) -> None:
        """Record token usage for LLM/VLM models."""
        total = input_tokens + output_tokens
        self.record(
            name="token_usage",
            value=total,
            unit="tokens",
            metric_type=MetricType.EFFICIENCY,
            metadata={
                "model": model_name,
                "task_id": task_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self) -> Dict[str, MetricAggregate]:
        """Compute aggregate statistics for all recorded metrics."""
        aggregates: Dict[str, MetricAggregate] = {}

        for name, metric_list in self.metrics.items():
            if not metric_list:
                continue

            values = [m.value for m in metric_list]
            unit = metric_list[0].unit
            metric_type = metric_list[0].metric_type

            agg = MetricAggregate(
                name=name,
                unit=unit,
                count=len(values),
                mean=statistics.mean(values),
                std=statistics.stdev(values) if len(values) > 1 else 0.0,
                min=min(values),
                max=max(values),
                median=statistics.median(values),
                q25=statistics.quantiles(values, n=4)[0] if len(values) >= 4 else values[0],
                q75=statistics.quantiles(values, n=4)[2] if len(values) >= 4 else values[-1],
                metric_type=metric_type,
            )
            aggregates[name] = agg

        return aggregates

    def summary(self) -> Dict[str, Any]:
        """Generate a high-level summary."""
        aggregates = self.aggregate()
        summary: Dict[str, Any] = {
            "total_metrics": sum(len(lst) for lst in self.metrics.values()),
            "unique_metrics": len(self.metrics),
            "aggregates": {k: v.to_dict() for k, v in aggregates.items()},
        }

        # Success rate
        if "task_success" in aggregates:
            success_agg = aggregates["task_success"]
            summary["success_rate"] = success_agg.mean
            summary["success_runs"] = int(success_agg.mean * success_agg.count)
            summary["total_runs"] = success_agg.count

        # Latency breakdown
        latency_keys = [k for k in aggregates if k.startswith("latency_")]
        if latency_keys:
            summary["latency"] = {}
            for key in latency_keys:
                agg = aggregates[key]
                summary["latency"][key] = {
                    "mean_ms": agg.mean,
                    "std_ms": agg.std,
                    "min_ms": agg.min,
                    "max_ms": agg.max,
                }

        # Resource usage
        resource_keys = [
            k for k in aggregates
            if aggregates[k].metric_type == MetricType.RESOURCE
        ]
        if resource_keys:
            summary["resources"] = {}
            for key in resource_keys:
                agg = aggregates[key]
                summary["resources"][key] = {
                    "mean": agg.mean,
                    "unit": agg.unit,
                }

        return summary

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(self, prefix: str = "") -> None:
        """Save all metrics to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = prefix + "_" if prefix else ""
        base_name = f"{prefix}metrics_{timestamp}"

        # Raw metrics
        raw_path = self.output_dir / f"{base_name}_raw.json"
        raw_data = []
        for metric_list in self.metrics.values():
            for metric in metric_list:
                raw_data.append(metric.to_dict())
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved raw metrics: {raw_path}")

        # Aggregates
        agg_path = self.output_dir / f"{base_name}_aggregates.json"
        aggregates = self.aggregate()
        agg_data = {k: v.to_dict() for k, v in aggregates.items()}
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved aggregates: {agg_path}")

        # Summary
        summary_path = self.output_dir / f"{base_name}_summary.json"
        summary = self.summary()
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved summary: {summary_path}")

        # CSV for easy analysis
        csv_path = self.output_dir / f"{base_name}.csv"
        rows = []
        for name, metric_list in self.metrics.items():
            for metric in metric_list:
                row = metric.to_dict()
                row["metric_name"] = name
                rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV: {csv_path}")

    # ------------------------------------------------------------------
    # Visualization (basic)
    # ------------------------------------------------------------------

    def plot_latency(self, save_path: Optional[Path] = None) -> None:
        """Generate a simple latency bar chart."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not installed, skipping plot")
            return

        latency_aggs = {
            k: v for k, v in self.aggregate().items()
            if k.startswith("latency_")
        }
        if not latency_aggs:
            logger.warning("No latency metrics to plot")
            return

        stages = [k.replace("latency_", "") for k in latency_aggs.keys()]
        means = [agg.mean for agg in latency_aggs.values()]
        stds = [agg.std for agg in latency_aggs.values()]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(stages, means, yerr=stds, capsize=5, color="skyblue")
        plt.xlabel("Stage")
        plt.ylabel("Latency (ms)")
        plt.title("Latency by Stage")
        plt.xticks(rotation=45)

        # Add value labels
        for bar, mean in zip(bars, means):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{mean:.1f}",
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

    def plot_success_by_model(self, save_path: Optional[Path] = None) -> None:
        """Plot success rate by model."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not installed, skipping plot")
            return

        # Group success metrics by model from metadata
        model_success: Dict[str, List[float]] = {}
        for metric_list in self.metrics.get("task_success", []):
            model = metric_list.metadata.get("model", "unknown")
            model_success.setdefault(model, []).append(metric_list.value)

        if not model_success:
            logger.warning("No success metrics to plot")
            return

        models = list(model_success.keys())
        success_rates = [statistics.mean(vals) * 100 for vals in model_success.values()]
        counts = [len(vals) for vals in model_success.values()]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, success_rates, color=["green" if r > 80 else "orange" for r in success_rates])
        plt.xlabel("Model")
        plt.ylabel("Success Rate (%)")
        plt.title("Success Rate by Model")
        plt.ylim(0, 105)

        # Add count labels
        for bar, rate, count in zip(bars, success_rates, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}% (n={count})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved success plot: {save_path}")
        else:
            plt.show()
        plt.close()
