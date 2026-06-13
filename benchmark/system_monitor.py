"""System resource monitor for benchmarking.

Tracks CPU, GPU, RAM, and disk usage in real-time
during GUI agent task execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ResourceSnapshot:
    """Single resource usage snapshot."""
    timestamp: float = 0.0
    cpu_percent: float = 0.0
    cpu_count: int = 0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    ram_percent: float = 0.0
    gpu_name: str = ""
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_load: float = 0.0
    gpu_temp: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0


@dataclass
class ResourceReport:
    """Aggregate resource report for a task run."""
    snapshots: List[ResourceSnapshot] = field(default_factory=list)
    peak_cpu_percent: float = 0.0
    peak_ram_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_ram_mb: float = 0.0
    avg_gpu_memory_mb: float = 0.0


class SystemMonitor:
    """Real-time system resource monitor."""

    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval = interval_sec
        self._snapshots: List[ResourceSnapshot] = []
        self._running = False
        self._prev_disk = None

    def snapshot(self) -> Dict[str, Any]:
        """Take a single resource snapshot and return as dict."""
        snap = self._capture()
        self._snapshots.append(snap)
        return self._snapshot_to_dict(snap)

    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        self._running = True

    def stop_monitoring(self) -> ResourceReport:
        """Stop monitoring and return report."""
        self._running = False
        # Take one final snapshot
        self._capture()
        return self._generate_report()

    def _capture(self) -> ResourceSnapshot:
        snap = ResourceSnapshot(timestamp=time.perf_counter())

        # CPU
        if HAS_PSUTIL:
            snap.cpu_percent = psutil.cpu_percent(interval=0.1)
            snap.cpu_count = psutil.cpu_count()

            # RAM
            mem = psutil.virtual_memory()
            snap.ram_used_mb = mem.used / (1024 ** 2)
            snap.ram_total_mb = mem.total / (1024 ** 2)
            snap.ram_percent = mem.percent

            # Disk I/O
            disk = psutil.disk_io_counters()
            if disk and self._prev_disk:
                snap.disk_read_mb = (disk.read_bytes - self._prev_disk.read_bytes) / (1024 ** 2)
                snap.disk_write_mb = (disk.write_bytes - self._prev_disk.write_bytes) / (1024 ** 2)
            self._prev_disk = disk

        # GPU via GPUtil
        if HAS_GPUTIL:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                snap.gpu_name = gpu.name
                snap.gpu_memory_used_mb = gpu.memoryUsed
                snap.gpu_memory_total_mb = gpu.memoryTotal
                snap.gpu_load = gpu.load * 100
                snap.gpu_temp = gpu.temperature

        # GPU via PyTorch
        if HAS_TORCH and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if not snap.gpu_name:
                snap.gpu_name = device_name
            mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            snap.gpu_memory_used_mb = max(snap.gpu_memory_used_mb, mem_allocated)
            snap.gpu_memory_total_mb = max(snap.gpu_memory_total_mb, total)

        return snap

    def _generate_report(self) -> ResourceReport:
        if not self._snapshots:
            return ResourceReport()

        cpu_values = [s.cpu_percent for s in self._snapshots if s.cpu_percent > 0]
        ram_values = [s.ram_used_mb for s in self._snapshots if s.ram_used_mb > 0]
        gpu_values = [s.gpu_memory_used_mb for s in self._snapshots if s.gpu_memory_used_mb > 0]

        return ResourceReport(
            snapshots=self._snapshots,
            peak_cpu_percent=max(cpu_values) if cpu_values else 0.0,
            peak_ram_mb=max(ram_values) if ram_values else 0.0,
            peak_gpu_memory_mb=max(gpu_values) if gpu_values else 0.0,
            avg_cpu_percent=sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
            avg_ram_mb=sum(ram_values) / len(ram_values) if ram_values else 0.0,
            avg_gpu_memory_mb=sum(gpu_values) / len(gpu_values) if gpu_values else 0.0,
        )

    @staticmethod
    def _snapshot_to_dict(snap: ResourceSnapshot) -> Dict[str, Any]:
        return {
            "timestamp": snap.timestamp,
            "cpu": {
                "percent": snap.cpu_percent,
                "count": snap.cpu_count,
                "brand": "Detected via psutil",
            },
            "ram": {
                "used_mb": snap.ram_used_mb,
                "total_mb": snap.ram_total_mb,
                "percent": snap.ram_percent,
            },
            "gpu": {
                "name": snap.gpu_name,
                "memory_used_mb": snap.gpu_memory_used_mb,
                "memory_total_mb": snap.gpu_memory_total_mb,
                "load_percent": snap.gpu_load,
                "temperature": snap.gpu_temp,
            },
            "disk": {
                "read_mb": snap.disk_read_mb,
                "write_mb": snap.disk_write_mb,
            },
        }

    def get_summary(self) -> str:
        """Human-readable system summary."""
        lines = []
        if HAS_PSUTIL:
            lines.append(f"CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count()} threads)")
            ram = psutil.virtual_memory()
            lines.append(f"RAM: {ram.total / (1024**3):.1f} GB total")
        if HAS_GPUTIL:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                lines.append(f"GPU: {gpu.name} ({gpu.memoryTotal}MB VRAM)")
        elif HAS_TORCH and torch.cuda.is_available():
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            lines.append(f"VRAM: {total:.1f} GB")
        return "\n".join(lines)
