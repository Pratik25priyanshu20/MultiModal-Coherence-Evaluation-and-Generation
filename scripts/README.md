# Scripts Guide

## Core Pipeline

| Script | Purpose |
|--------|---------|
| `build_embedding_indexes.py` | Build CLIP/CLAP embedding indexes from raw data |
| `run_full_evaluation.py` | Run complete evaluation (cMSCI + baselines + ablation) |
| `generate_paper_figures.py` | Generate all paper figures (fig1–20) |
| `generate_architecture_diagram.py` | Generate system architecture diagram |

## Training

| Script | Purpose |
|--------|---------|
| `train_bridge.py` | Train cross-space bridge (CLIP↔CLAP) |
| `train_exmcr.py` | Train Ex-MCR projector for complementarity |
| `train_cmsci_models.py` | Train all cMSCI sub-models end-to-end |
| `prepare_bridge_data.py` | Prepare paired training data for bridge |

## cMSCI Optimization & Ablation

| Script | Purpose |
|--------|---------|
| `optimize_cmsci.py` | Hyperparameter optimization (dev set only) |
| `run_cmsci_ablation.py` | Ablation study (MSCI → Gram → z-score → contrastive → Ex-MCR → full) |
| `run_cmsci_comparison.py` | Compare cMSCI variants A–F |
| `extend_calibration.py` | Extend calibration data for z-score normalization |

## Robustness & Sensitivity

| Script | Purpose |
|--------|---------|
| `run_seed_robustness.py` | Test stability across 10 random seeds |
| `run_sensitivity.py` | Hyperparameter sensitivity analysis (alpha, w_ti, w_3d, gamma) |
| `run_negbank_robustness.py` | Negative bank size/composition robustness |
| `analyze_failures.py` | Failure case analysis with error categorization |

## Baselines & Benchmarks

| Script | Purpose |
|--------|---------|
| `evaluate_benchmarks.py` | Run AudioCaps/VGGSound benchmarks |
| `download_benchmarks.py` | Download benchmark datasets |
| `analyze_clap_sensitivity.py` | CLAP audio sensitivity analysis |

## Experiments (RQ1–RQ3)

| Script | Purpose |
|--------|---------|
| `run_rq1.py` / `run_rq1_hybrid.py` | RQ1: MSCI sensitivity (generation / hybrid) |
| `run_rq2.py` / `run_rq2_hybrid.py` | RQ2: Planning ablation (generation / hybrid) |
| `run_rq3.py` → `analyze_rq3.py` | RQ3: Human evaluation analysis |
| `run_human_eval.py` | Launch CLI human evaluation session |
| `run_controlled_experiment.py` | Controlled experiment runner |
| `analyze_results.py` | General result analysis |
| `analyze_controlled_experiment.py` | RQ1/RQ2 statistical analysis |

## Data Preparation

| Script | Purpose |
|--------|---------|
| `build_wikimedia_dataset.py` | Download Wikimedia images/audio |
| `build_audiocaps_subset.py` | Download AudioCaps subset |
| `build_laion_subset.py` | Download LAION subset |
| `build_freesound_dataset.py` | Download Freesound audio |
| `build_gold_pilot.py` | Build gold-standard pilot set |
| `select_rq3_samples.py` | Select stratified samples for RQ3 |
| `generate_new_eval_samples.py` | Generate new evaluation samples |

## Figures

| Script | Purpose |
|--------|---------|
| `generate_paper_figures.py` | Main paper figures (fig1–20) |
| `generate_final_figures.py` | Publication-quality final figures |
| `generate_architecture_diagram.py` | System architecture diagram |
| `visualize_embeddings.py` | t-SNE/UMAP/PCA embedding visualizations |

## Deployment

| Script | Purpose |
|--------|---------|
| `deploy_hf.sh` | Deploy to HuggingFace Spaces |

## Validation & Diagnostics

| Script | Purpose |
|--------|---------|
| `validate_msci.py` | Validate MSCI computation correctness |
| `sanity_check.py` | Full pipeline sanity check |
| `diagnose_retrieval.py` | Debug retrieval issues |
| `verify_conditioning.py` | Check planner conditioning |
| `calibrate_metrics.py` | Calibrate adaptive thresholds |
| `fit_coherence_stats.py` | Fit coherence distribution statistics |

## Legacy (can be ignored)

These are superseded by newer scripts but kept for reproducibility:

`batch_run_phase2.py`, `run_phase2_v1.py`, `apply_fixes_and_test.py`,
`audioset_to_soundscape_text.py`, `test_embeddings.py`, `test_semantic_planner.py`,
`test_optimizations.py`, `check_optimizations.py`, `run_unified.py`,
`run_unified_batch.py`, `analyze_unified_batch.py`, `run_perturbation.py`,
`run_ablation_study.py`, `run_dataset_eval.py`, `run_laion_eval.py`,
`analyze_laion_failures.py`, `analyze_failure_modes.py`,
`analyze_retrieval_bottleneck.py`, `summarize_dataset_eval.py`,
`summarize_runs.py`, `evaluate_with_bridge.py`, `regenerate_rq3_audio.py`,
`run_full_experiment.py`, `prompts_batch.json`
