"""
Performance monitoring utilities for tracking inference time, throughput, and memory usage.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import psutil
import torch


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""

    inference_time: float = 0.0  # seconds
    memory_used_mb: float = 0.0  # megabytes
    throughput: float = 0.0  # items per second
    batch_size: int = 1
    device: str = "cpu"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""

    total_calls: int = 0
    total_time: float = 0.0
    total_memory: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    avg_time: float = 0.0
    min_memory: float = float("inf")
    max_memory: float = 0.0
    avg_memory: float = 0.0
    avg_throughput: float = 0.0


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        self.metrics: list[PerformanceMetrics] = []
        self._stats: Dict[str, PerformanceStats] = {}

    def record(
        self,
        inference_time: float,
        memory_used_mb: float = 0.0,
        batch_size: int = 1,
        device: str = "cpu",
        metadata: Optional[Dict[str, Any]] = None,
        operation_name: str = "operation",
    ) -> PerformanceMetrics:
        """Record performance metrics."""
        throughput = batch_size / inference_time if inference_time > 0 else 0.0
        
        metric = PerformanceMetrics(
            inference_time=inference_time,
            memory_used_mb=memory_used_mb,
            throughput=throughput,
            batch_size=batch_size,
            device=device,
            metadata=metadata or {},
        )
        
        self.metrics.append(metric)
        
        # Update stats
        if operation_name not in self._stats:
            self._stats[operation_name] = PerformanceStats()
        
        stats = self._stats[operation_name]
        stats.total_calls += 1
        stats.total_time += inference_time
        stats.total_memory += memory_used_mb
        stats.min_time = min(stats.min_time, inference_time)
        stats.max_time = max(stats.max_time, inference_time)
        stats.min_memory = min(stats.min_memory, memory_used_mb)
        stats.max_memory = max(stats.max_memory, memory_used_mb)
        stats.avg_time = stats.total_time / stats.total_calls
        stats.avg_memory = stats.total_memory / stats.total_calls
        stats.avg_throughput = batch_size / stats.avg_time if stats.avg_time > 0 else 0.0
        
        return metric

    def get_stats(self, operation_name: Optional[str] = None) -> Dict[str, PerformanceStats]:
        """Get performance statistics."""
        if operation_name:
            return {operation_name: self._stats.get(operation_name, PerformanceStats())}
        return self._stats.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all performance metrics."""
        summary = {}
        for op_name, stats in self._stats.items():
            summary[op_name] = {
                "total_calls": stats.total_calls,
                "avg_time_seconds": stats.avg_time,
                "min_time_seconds": stats.min_time,
                "max_time_seconds": stats.max_time,
                "avg_memory_mb": stats.avg_memory,
                "min_memory_mb": stats.min_memory,
                "max_memory_mb": stats.max_memory,
                "avg_throughput": stats.avg_throughput,
            }
        return summary

    def reset(self) -> None:
        """Reset all metrics and statistics."""
        self.metrics.clear()
        self._stats.clear()


def get_memory_usage_mb(process: Optional[psutil.Process] = None) -> float:
    """Get current memory usage in MB."""
    if process is None:
        process = psutil.Process()
    try:
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def get_gpu_memory_mb(device: str = "cuda:0") -> float:
    """Get GPU memory usage in MB."""
    try:
        if torch.cuda.is_available() and device.startswith("cuda"):
            device_id = int(device.split(":")[1]) if ":" in device else 0
            return torch.cuda.memory_allocated(device_id) / 1024 / 1024
    except Exception:
        pass
    return 0.0


@contextmanager
def measure_performance(
    monitor: PerformanceMonitor,
    operation_name: str = "operation",
    batch_size: int = 1,
    device: str = "cpu",
    metadata: Optional[Dict[str, Any]] = None,
):
    """Context manager to measure performance of a code block."""
    process = psutil.Process()
    memory_before = get_memory_usage_mb(process)
    
    if device.startswith("cuda"):
        gpu_memory_before = get_gpu_memory_mb(device)
    else:
        gpu_memory_before = 0.0
    
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        inference_time = end_time - start_time
        
        memory_after = get_memory_usage_mb(process)
        memory_used = memory_after - memory_before
        
        if device.startswith("cuda"):
            gpu_memory_after = get_gpu_memory_mb(device)
            gpu_memory_used = gpu_memory_after - gpu_memory_before
            memory_used = max(memory_used, gpu_memory_used)
        
        monitor.record(
            inference_time=inference_time,
            memory_used_mb=max(memory_used, 0.0),  # Can be negative due to garbage collection
            batch_size=batch_size,
            device=device,
            metadata=metadata or {},
            operation_name=operation_name,
        )


def monitor_performance(
    operation_name: Optional[str] = None,
    batch_size: int = 1,
    device: str = "cpu",
    monitor: Optional[PerformanceMonitor] = None,
):
    """Decorator to monitor performance of a function."""
    if monitor is None:
        monitor = PerformanceMonitor()

    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with measure_performance(monitor, operation_name=name, batch_size=batch_size, device=device):
                return func(*args, **kwargs)

        wrapper._monitor = monitor  # Attach monitor to function
        return wrapper

    return decorator
