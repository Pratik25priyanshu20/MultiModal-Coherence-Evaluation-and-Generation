"""
Real-time evaluation module for streaming data and generating outputs on-the-fly.

Supports:
- Streaming evaluation
- Real-time monitoring
- Progressive result aggregation
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from src.utils.performance_monitor import PerformanceMonitor


@dataclass
class RealtimeMetric:
    """Single metric update in real-time evaluation."""

    timestamp: float
    sample_id: str
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealtimeResult:
    """Result from real-time evaluation."""

    sample_id: str
    timestamp: float
    scores: Dict[str, float]
    coherence: Dict[str, Any]
    performance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealtimeEvaluator:
    """Real-time evaluator for streaming evaluation."""

    def __init__(
        self,
        evaluation_func: Callable[[Any], Dict[str, Any]],
        output_dir: Optional[str] = None,
        enable_monitoring: bool = True,
    ):
        self.evaluation_func = evaluation_func
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_monitoring = enable_monitoring
        self.monitor = PerformanceMonitor() if enable_monitoring else None
        
        self.results: List[RealtimeResult] = []
        self.metrics: List[RealtimeMetric] = []
        self._start_time = time.time()

    def evaluate_stream(
        self,
        samples: Iterator[Any],
        sample_id_func: Optional[Callable[[Any], str]] = None,
    ) -> Iterator[RealtimeResult]:
        """
        Evaluate samples in a stream, yielding results as they become available.
        
        Args:
            samples: Iterator of samples to evaluate
            sample_id_func: Function to extract sample ID from sample
        
        Yields:
            RealtimeResult for each evaluated sample
        """
        for idx, sample in enumerate(samples):
            sample_id = sample_id_func(sample) if sample_id_func else f"sample_{idx}"
            
            # Evaluate sample
            start_time = time.time()
            
            if self.enable_monitoring and self.monitor:
                from src.utils.performance_monitor import measure_performance
                with measure_performance(
                    self.monitor,
                    operation_name="realtime_evaluation",
                    batch_size=1,
                    metadata={"sample_id": sample_id},
                ):
                    result_data = self.evaluation_func(sample)
            else:
                result_data = self.evaluation_func(sample)
            
            eval_time = time.time() - start_time
            
            # Extract scores and coherence
            scores = result_data.get("scores", {})
            coherence = result_data.get("coherence", {})
            
            # Get performance metrics if available
            performance = {}
            if self.enable_monitoring and self.monitor:
                stats = self.monitor.get_stats("realtime_evaluation")
                if stats:
                    perf_stats = stats.get("realtime_evaluation")
                    if perf_stats:
                        performance = {
                            "inference_time": perf_stats.avg_time,
                            "throughput": perf_stats.avg_throughput,
                        }
            
            # Create result
            result = RealtimeResult(
                sample_id=sample_id,
                timestamp=time.time(),
                scores=scores,
                coherence=coherence,
                performance=performance,
                metadata=result_data.get("metadata", {}),
            )
            
            self.results.append(result)
            
            # Emit metrics
            for metric_name, value in scores.items():
                metric = RealtimeMetric(
                    timestamp=time.time(),
                    sample_id=sample_id,
                    metric_name=metric_name,
                    value=value,
                )
                self.metrics.append(metric)
            
            # Save result if output directory is set
            if self.output_dir:
                self._save_result(result)
            
            yield result

    def evaluate_batch(
        self,
        samples: List[Any],
        sample_id_func: Optional[Callable[[Any], str]] = None,
    ) -> List[RealtimeResult]:
        """
        Evaluate a batch of samples, returning results.
        
        Args:
            samples: List of samples to evaluate
            sample_id_func: Function to extract sample ID from sample
        
        Returns:
            List of RealtimeResult
        """
        results = list(self.evaluate_stream(iter(samples), sample_id_func=sample_id_func))
        return results

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics from evaluated results."""
        if not self.results:
            return {}
        
        # Aggregate scores
        all_scores: Dict[str, List[float]] = {}
        for result in self.results:
            for metric_name, value in result.scores.items():
                if metric_name not in all_scores:
                    all_scores[metric_name] = []
                all_scores[metric_name].append(value)
        
        # Compute statistics
        stats = {}
        for metric_name, values in all_scores.items():
            import numpy as np
            stats[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }
        
        # Overall statistics
        stats["total_samples"] = len(self.results)
        stats["total_time"] = time.time() - self._start_time
        stats["avg_throughput"] = len(self.results) / stats["total_time"] if stats["total_time"] > 0 else 0.0
        
        return stats

    def get_metrics_history(self, metric_name: Optional[str] = None) -> List[RealtimeMetric]:
        """Get history of metrics."""
        if metric_name:
            return [m for m in self.metrics if m.metric_name == metric_name]
        return self.metrics.copy()

    def _save_result(self, result: RealtimeResult) -> None:
        """Save individual result to disk."""
        if not self.output_dir:
            return
        
        result_file = self.output_dir / f"{result.sample_id}.json"
        with result_file.open("w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

    def save_summary(self, output_path: Optional[str] = None) -> None:
        """Save evaluation summary to disk."""
        if output_path is None and self.output_dir:
            output_path = str(self.output_dir / "summary.json")
        
        if output_path is None:
            return
        
        summary = {
            "aggregate_stats": self.get_aggregate_stats(),
            "total_results": len(self.results),
            "performance_summary": self.monitor.get_summary() if self.monitor else {},
        }
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def reset(self) -> None:
        """Reset evaluator state."""
        self.results.clear()
        self.metrics.clear()
        self._start_time = time.time()
        if self.monitor:
            self.monitor.reset()
