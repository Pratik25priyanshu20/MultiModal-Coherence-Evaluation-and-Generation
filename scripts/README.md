# Scripts Guide

## Which Script Should I Use?

### Running Experiments

| Script | Purpose | When to use |
|--------|---------|-------------|
| `run_unified.py` | Run a single prompt through the pipeline | Quick testing / debugging |
| `run_rq1.py` | RQ1: MSCI sensitivity experiment (270 runs) | Full RQ1 experiment |
| `run_rq2.py` | RQ2: Planning ablation study | Full RQ2 experiment |
| `run_human_eval.py` | RQ3: Launch CLI human evaluation | Prefer the Streamlit app instead |
| `run_full_experiment.py` | Run all experiments end-to-end | Comprehensive batch run |
| `sanity_check.py` | Validate pipeline correctness | After any code/data change |

### Analyzing Results

| Script | Purpose | When to use |
|--------|---------|-------------|
| `analyze_results.py` | General result analysis | After any experiment run |
| `analyze_controlled_experiment.py` | RQ1/RQ2 statistical analysis | After RQ1 or RQ2 |
| `analyze_rq3.py` | RQ3 inter-rater analysis | After human evaluation sessions |
| `analyze_retrieval_bottleneck.py` | Diagnose retrieval quality | If retrieval seems poor |

### Data Preparation

| Script | Purpose | When to use |
|--------|---------|-------------|
| `build_embedding_indexes.py` | Build CLIP/CLAP embedding indexes | After adding/changing data |
| `build_wikimedia_dataset.py` | Download Wikimedia images/audio | One-time data setup |
| `build_audiocaps_subset.py` | Download AudioCaps subset | One-time data setup |
| `select_rq3_samples.py` | Select stratified samples for RQ3 | One-time RQ3 setup |

### Validation & Diagnostics

| Script | Purpose | When to use |
|--------|---------|-------------|
| `validate_msci.py` | Validate MSCI computation | After embedding changes |
| `diagnose_retrieval.py` | Debug retrieval issues | Troubleshooting |
| `verify_conditioning.py` | Check planner conditioning | After planner changes |
| `fit_coherence_stats.py` | Calibrate adaptive thresholds | After major data changes |

### Legacy / One-off (can be ignored)

`batch_run_phase2.py`, `run_phase2_v1.py`, `apply_fixes_and_test.py`,
`audioset_to_soundscape_text.py`, `test_embeddings.py`, `test_semantic_planner.py`,
`test_optimizations.py`, `check_optimizations.py` — superseded by newer scripts.
