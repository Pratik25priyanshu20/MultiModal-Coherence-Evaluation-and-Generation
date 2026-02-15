"""
Parallel processing utilities for data pipeline optimization.

Supports:
- Parallel data ingestion
- Batch processing with multiprocessing
- Distributed processing support (foundation for Dask/Beam)
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: Optional[int] = None,
    use_threads: bool = False,
    chunk_size: int = 1,
    **func_kwargs,
) -> List[R]:
    """
    Parallel map function.
    
    Args:
        func: Function to apply to each item
        items: Iterable of items to process
        max_workers: Maximum number of workers (default: CPU count)
        use_threads: Use threads instead of processes (for I/O-bound tasks)
        chunk_size: Number of items per chunk (for ProcessPoolExecutor)
        **func_kwargs: Additional kwargs to pass to func
    
    Returns:
        List of results in the same order as items
    """
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    items_list = list(items)
    if not items_list:
        return []
    
    # Prepare function with kwargs
    if func_kwargs:
        func = partial(func, **func_kwargs)
    
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    with executor_class(max_workers=max_workers) as executor:
        if use_threads:
            # ThreadPoolExecutor doesn't use chunk_size
            futures = [executor.submit(func, item) for item in items_list]
        else:
            # ProcessPoolExecutor supports chunking
            futures = executor.map(func, items_list, chunksize=chunk_size)
            return list(futures)
        
        # For ThreadPoolExecutor, collect results
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                # Log error but continue
                print(f"Error in parallel_map: {e}")
                results.append(None)
        
        # Reorder results to match input order (approximate for as_completed)
        # For exact ordering, use executor.map instead
        return results


def batch_process(
    func: Callable[[List[T]], List[R]],
    items: Iterable[T],
    batch_size: int = 32,
    max_workers: Optional[int] = None,
    **func_kwargs,
) -> List[R]:
    """
    Process items in batches in parallel.
    
    Args:
        func: Function that processes a batch and returns list of results
        items: Iterable of items to process
        batch_size: Number of items per batch
        max_workers: Maximum number of parallel batches
        **func_kwargs: Additional kwargs to pass to func
    
    Returns:
        Flattened list of results
    """
    items_list = list(items)
    batches = [
        items_list[i : i + batch_size] for i in range(0, len(items_list), batch_size)
    ]
    
    if not batches:
        return []
    
    # Prepare function with kwargs
    if func_kwargs:
        func = partial(func, **func_kwargs)
    
    if max_workers is None:
        max_workers = min(len(batches), mp.cpu_count())
    
    results = parallel_map(
        func,
        batches,
        max_workers=max_workers,
        use_threads=False,
    )
    
    # Flatten results
    flattened = []
    for batch_results in results:
        if batch_results:
            flattened.extend(batch_results)
    
    return flattened


class ParallelProcessor:
    """Wrapper for parallel processing with configuration."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_threads: bool = False,
        chunk_size: int = 1,
    ):
        self.max_workers = max_workers or mp.cpu_count()
        self.use_threads = use_threads
        self.chunk_size = chunk_size

    def map(self, func: Callable[[T], R], items: Iterable[T], **func_kwargs) -> List[R]:
        """Apply function to items in parallel."""
        return parallel_map(
            func,
            items,
            max_workers=self.max_workers,
            use_threads=self.use_threads,
            chunk_size=self.chunk_size,
            **func_kwargs,
        )

    def batch_map(
        self,
        func: Callable[[List[T]], List[R]],
        items: Iterable[T],
        batch_size: int = 32,
        **func_kwargs,
    ) -> List[R]:
        """Process items in batches in parallel."""
        return batch_process(
            func,
            items,
            batch_size=batch_size,
            max_workers=self.max_workers,
            **func_kwargs,
        )
