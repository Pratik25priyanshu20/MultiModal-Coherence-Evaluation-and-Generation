# Quick Start Guide: Optimizations

This guide shows you how to quickly enable and use the newly implemented optimizations.

## 1. Enable Caching (Automatic)

Caching is **enabled by default** in `AlignedEmbedder`. No changes needed!

If you want to customize:
```python
from src.embeddings.aligned_embeddings import AlignedEmbedder

# Custom cache directory
embedder = AlignedEmbedder(
    enable_cache=True,
    cache_dir=".cache/custom_embeddings"
)

# Disable caching (for debugging)
embedder = AlignedEmbedder(enable_cache=False)
```

## 2. Add Performance Monitoring

Add monitoring to any function:

```python
from src.utils.performance_monitor import PerformanceMonitor, measure_performance

monitor = PerformanceMonitor()

# Method 1: Context manager
with measure_performance(monitor, "image_generation", batch_size=1):
    image_path = image_gen.generate(prompt)

# Method 2: Decorator
from src.utils.performance_monitor import monitor_performance

@monitor_performance(operation_name="embedding", batch_size=1)
def my_embedding_function(text):
    return embedder.embed_text(text)

# Get stats
stats = monitor.get_stats("image_generation")
summary = monitor.get_summary()
print(f"Average time: {stats['image_generation'].avg_time_seconds:.3f}s")
```

## 3. Use Dynamic Council

Replace the standard council with the dynamic council:

```python
from src.planner.dynamic_council import DynamicSemanticCouncil
from src.planner.unified_planner import UnifiedPlannerLLM
from src.orchestrator.request_flow import Orchestrator

# Create planners
planner_a = UnifiedPlannerLLM()
planner_b = UnifiedPlannerLLM()
planner_c = UnifiedPlannerLLM()

# Create dynamic council
dynamic_council = DynamicSemanticCouncil(
    planner_a=planner_a,
    planner_b=planner_b,
    planner_c=planner_c,
    enable_dynamic_weighting=True,  # Enable dynamic weighting
)

# Use in orchestrator
orchestrator = Orchestrator(
    council=dynamic_council,  # Use dynamic council
    text_gen=text_gen,
    image_gen=image_gen,
    audio_gen=audio_gen,
)

# Get modality priority for a prompt
priority = dynamic_council.get_modality_priority("A beautiful sunset with birds singing")
print(f"Primary modality: {priority.primary}")
print(f"Weights: text={priority.weights.text_weight:.2f}, "
      f"image={priority.weights.image_weight:.2f}, "
      f"audio={priority.weights.audio_weight:.2f}")
```

## 4. Parallel Processing

Process data in parallel:

```python
from src.utils.parallel_processing import ParallelProcessor

# Create processor
processor = ParallelProcessor(max_workers=4, use_threads=False)

# Process items in parallel
def process_item(item):
    # Your processing logic
    return processed_result

results = processor.map(process_item, items)

# Batch processing
def process_batch(batch):
    return [process_item(item) for item in batch]

batch_results = processor.batch_map(process_batch, items, batch_size=32)
```

## 5. Real-Time Evaluation

Evaluate samples in real-time:

```python
from src.evaluation.realtime_evaluator import RealtimeEvaluator

# Define evaluation function
def evaluate_sample(sample):
    # Your evaluation logic
    result = orchestrator.run(sample["prompt"])
    return {
        "scores": result.scores,
        "coherence": result.coherence,
        "metadata": {"run_id": result.run_id},
    }

# Create evaluator
evaluator = RealtimeEvaluator(
    evaluation_func=evaluate_sample,
    output_dir="runs/realtime",
    enable_monitoring=True,
)

# Stream evaluation
samples = [{"prompt": p} for p in prompts]
for result in evaluator.evaluate_stream(iter(samples)):
    print(f"Sample {result.sample_id}: MSCI={result.scores.get('msci', 0):.4f}")

# Get summary
summary = evaluator.get_aggregate_stats()
print(f"Average MSCI: {summary.get('msci', {}).get('mean', 0):.4f}")
print(f"Throughput: {summary.get('avg_throughput', 0):.2f} samples/sec")

# Save summary
evaluator.save_summary("runs/realtime_summary.json")
```

## 6. Audio Enhancement (Foundation)

Use audio enhancement utilities:

```python
from src.generators.audio.enhancement import AudioEnhancer, AudioAlignmentImprover

# Audio analysis and enhancement
enhancer = AudioEnhancer(sample_rate=48000)

# Analyze audio
analysis = enhancer.analyze_audio("path/to/audio.wav")
print(f"Duration: {analysis.duration:.2f}s")
print(f"Segments: {len(analysis.segments)}")

# Segment audio
segments = enhancer.segment_audio("path/to/audio.wav", segment_duration=2.0)

# Separate background/foreground
background, foreground = enhancer.separate_background_foreground("path/to/audio.wav")

# Improve temporal coherence
enhanced_audio = enhancer.improve_temporal_coherence("path/to/audio.wav")
sf.write("enhanced_audio.wav", enhanced_audio, 48000)

# Alignment improvement (placeholder for future models)
alignment_improver = AudioAlignmentImprover()
score = alignment_improver.compute_alignment_score(
    "path/to/audio.wav",
    "A quiet forest with birds chirping"
)
```

## Integration Examples

### Example 1: Add Monitoring to Existing Script

```python
# scripts/run_unified.py (example modification)
from src.utils.performance_monitor import PerformanceMonitor, measure_performance

monitor = PerformanceMonitor()

def main():
    # ... existing setup ...
    
    with measure_performance(monitor, "full_pipeline", batch_size=1):
        result = orchestrator.run(prompt)
    
    # Print performance summary
    summary = monitor.get_summary()
    print("\n=== Performance Summary ===")
    for op, stats in summary.items():
        print(f"{op}: {stats['avg_time_seconds']:.3f}s avg")
```

### Example 2: Use Dynamic Council in Orchestrator

```python
# scripts/run_phase2_v1.py (example modification)
from src.planner.dynamic_council import DynamicSemanticCouncil
from src.planner.unified_planner import UnifiedPlannerLLM

def main():
    # Replace standard council
    planner_a = UnifiedPlannerLLM()
    planner_b = UnifiedPlannerLLM()
    planner_c = UnifiedPlannerLLM()
    
    council = DynamicSemanticCouncil(
        planner_a, planner_b, planner_c,
        enable_dynamic_weighting=True,
    )
    
    # Rest of code remains the same
    orchestrator = Orchestrator(council=council, ...)
```

### Example 3: Parallel Dataset Processing

```python
# scripts/run_dataset_eval.py (example modification)
from src.utils.parallel_processing import ParallelProcessor

def process_sample(sample):
    # Your processing logic
    return result

def main():
    # Load samples
    samples = load_samples()
    
    # Process in parallel
    processor = ParallelProcessor(max_workers=4)
    results = processor.map(process_sample, samples)
    
    # Process results
    for result in results:
        save_result(result)
```

## Performance Tips

1. **Caching**: First run creates cache, subsequent runs are faster
2. **Parallel Processing**: Use `max_workers=CPU_count` for CPU-bound tasks, `use_threads=True` for I/O-bound
3. **Dynamic Council**: Best for prompts with clear modality emphasis (e.g., "A beautiful sunset" = image-primary)
4. **Real-Time Evaluation**: Use for long-running experiments to get immediate feedback
5. **Performance Monitoring**: Add to identify bottlenecks, then optimize

## Troubleshooting

### Cache Issues
- Clear cache: `rm -rf .cache/embeddings`
- Check cache size: `du -sh .cache/embeddings`

### Performance Monitoring
- Ensure `psutil` is installed: `pip install psutil`
- GPU monitoring requires CUDA-enabled PyTorch

### Parallel Processing
- Reduce `max_workers` if you see memory issues
- Use `use_threads=True` for I/O-bound tasks (file I/O, network)

## Next Steps

1. Test optimizations with your existing pipelines
2. Measure performance improvements
3. Integrate gradually (start with caching, then add monitoring)
4. See `OPTIMIZATION_IMPLEMENTATION.md` for detailed documentation
