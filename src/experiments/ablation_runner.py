"""
Ablation Runner Module

Runs ablation studies to definitively answer:
"Is Council-Lite's benefit from structure or just more prompting/tokens?"

Four experimental conditions:
1. Direct: Raw prompt → generators (no planning)
2. Single Planner: 1 LLM call → plan → generators
3. Council-Lite: 3 LLM calls → merge → generators
4. Extended Prompt: 1 LLM call with 3× token budget (controls for compute)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

from src.planner.single_planner import SinglePlanner, PlannerMetrics
from src.planner.extended_prompt_planner import ExtendedPromptPlanner
from src.planner.unified_planner import UnifiedPlanner
from src.planner.schema import SemanticPlan
from src.utils.seed import set_global_seed


@dataclass
class AblationCondition:
    """Definition of an ablation condition."""
    name: str
    description: str
    planner_class: Optional[type]
    expected_llm_calls: int
    token_multiplier: float = 1.0

    def create_planner(self, **kwargs):
        """Create planner instance for this condition."""
        if self.planner_class is None:
            return None  # Direct mode
        return self.planner_class(**kwargs)


# Define the four ablation conditions
ABLATION_CONDITIONS = {
    "direct": AblationCondition(
        name="direct",
        description="Raw prompt → generators (no planning)",
        planner_class=None,
        expected_llm_calls=0,
        token_multiplier=0.0,
    ),
    "single_planner": AblationCondition(
        name="single_planner",
        description="1 LLM call → plan → generators",
        planner_class=SinglePlanner,
        expected_llm_calls=1,
        token_multiplier=1.0,
    ),
    "council": AblationCondition(
        name="council",
        description="3 LLM calls (council) → merge → generators",
        planner_class=UnifiedPlanner,  # Uses council under the hood
        expected_llm_calls=3,
        token_multiplier=3.0,
    ),
    "extended_prompt": AblationCondition(
        name="extended_prompt",
        description="1 LLM call with 3× token budget",
        planner_class=ExtendedPromptPlanner,
        expected_llm_calls=1,
        token_multiplier=3.0,
    ),
}


@dataclass
class AblationResult:
    """Result from a single ablation run."""
    condition: str
    prompt: str
    seed: int
    success: bool
    msci: Optional[float] = None
    st_i: Optional[float] = None
    st_a: Optional[float] = None
    si_a: Optional[float] = None
    planner_metrics: Optional[Dict[str, Any]] = None
    generation_time_ms: float = 0.0
    error: Optional[str] = None
    run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition": self.condition,
            "prompt": self.prompt,
            "seed": self.seed,
            "success": self.success,
            "msci": self.msci,
            "st_i": self.st_i,
            "st_a": self.st_a,
            "si_a": self.si_a,
            "planner_metrics": self.planner_metrics,
            "generation_time_ms": self.generation_time_ms,
            "error": self.error,
            "run_id": self.run_id,
        }


@dataclass
class AblationStudyConfig:
    """Configuration for ablation study."""
    name: str = "council_lite_ablation"
    conditions: List[str] = field(default_factory=lambda: list(ABLATION_CONDITIONS.keys()))
    n_prompts: int = 50
    n_seeds: int = 3
    base_seed: int = 42
    output_dir: str = "runs/ablation_study"
    use_ollama: bool = True
    deterministic: bool = True

    @property
    def total_runs(self) -> int:
        """Total number of runs."""
        return self.n_prompts * self.n_seeds * len(self.conditions)


class AblationRunner:
    """
    Runs ablation studies across planning conditions.

    Key controls:
    - Same prompts across all conditions
    - Same seeds for reproducibility
    - Token budget tracking
    - Compute time tracking
    """

    def __init__(self, config: AblationStudyConfig):
        self.config = config
        self.results: List[AblationResult] = []
        self.results_by_condition: Dict[str, List[AblationResult]] = defaultdict(list)

    def run_single(
        self,
        prompt: str,
        condition: str,
        seed: int,
        output_dir: Path,
    ) -> AblationResult:
        """
        Run a single ablation condition.

        Args:
            prompt: Input prompt
            condition: Condition name from ABLATION_CONDITIONS
            seed: Random seed
            output_dir: Output directory for this run

        Returns:
            AblationResult
        """
        from src.pipeline.generate_and_evaluate import generate_and_evaluate

        if condition not in ABLATION_CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")

        cond_def = ABLATION_CONDITIONS[condition]

        if self.config.deterministic:
            set_global_seed(seed)

        start_time = time.time()
        planner_metrics = None

        try:
            # For direct mode, use the existing direct mode
            mode = "direct" if condition == "direct" else "planner"

            # Create planner if needed
            if cond_def.planner_class:
                planner = cond_def.create_planner()
                # Generate plan to get metrics
                plan = planner.plan(prompt)
                if hasattr(planner, 'get_metrics'):
                    metrics = planner.get_metrics()
                    if metrics:
                        planner_metrics = metrics.to_dict()

            # Run generation and evaluation
            bundle = generate_and_evaluate(
                prompt=prompt,
                out_dir=str(output_dir),
                use_ollama=self.config.use_ollama,
                deterministic=self.config.deterministic,
                seed=seed,
                mode=mode,
                condition="baseline",
            )

            end_time = time.time()

            return AblationResult(
                condition=condition,
                prompt=prompt,
                seed=seed,
                success=True,
                msci=bundle.scores.get("msci"),
                st_i=bundle.scores.get("st_i"),
                st_a=bundle.scores.get("st_a"),
                si_a=bundle.scores.get("si_a"),
                planner_metrics=planner_metrics,
                generation_time_ms=(end_time - start_time) * 1000,
                run_id=bundle.run_id,
            )

        except Exception as e:
            end_time = time.time()
            return AblationResult(
                condition=condition,
                prompt=prompt,
                seed=seed,
                success=False,
                error=str(e),
                generation_time_ms=(end_time - start_time) * 1000,
            )

    def run_study(
        self,
        prompts: List[str],
    ) -> Dict[str, Any]:
        """
        Run complete ablation study.

        Args:
            prompts: List of prompts to test

        Returns:
            Complete study results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path(self.config.output_dir) / f"{self.config.name}_{timestamp}"
        output_base.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"ABLATION STUDY: {self.config.name}")
        print(f"{'=' * 70}")
        print(f"Conditions: {self.config.conditions}")
        print(f"Prompts: {len(prompts)}")
        print(f"Seeds per prompt: {self.config.n_seeds}")
        print(f"Total runs: {self.config.total_runs}")
        print(f"Output: {output_base}")
        print(f"{'=' * 70}\n")

        # Run all conditions
        for prompt_idx, prompt in enumerate(prompts[:self.config.n_prompts]):
            print(f"\nPrompt {prompt_idx + 1}/{self.config.n_prompts}: {prompt[:50]}...")

            for seed_offset in range(self.config.n_seeds):
                seed = self.config.base_seed + prompt_idx * 100 + seed_offset

                for condition in self.config.conditions:
                    print(f"  [{condition}] seed={seed}...", end=" ")

                    result = self.run_single(
                        prompt=prompt,
                        condition=condition,
                        seed=seed,
                        output_dir=output_base / condition / f"prompt_{prompt_idx}_seed_{seed}",
                    )

                    self.results.append(result)
                    self.results_by_condition[condition].append(result)

                    if result.success:
                        print(f"MSCI={result.msci:.4f}")
                    else:
                        print(f"ERROR: {result.error[:40] if result.error else 'Unknown'}")

        # Generate report
        report = self._generate_report(timestamp, prompts)

        # Save results
        results_path = output_base / "ablation_results.json"
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nResults saved to: {results_path}")
        self._print_summary(report)

        return report

    def _generate_report(
        self,
        timestamp: str,
        prompts: List[str],
    ) -> Dict[str, Any]:
        """Generate comprehensive ablation report."""
        import numpy as np
        from src.experiments.statistical_analysis import (
            paired_ttest,
            compare_all_pairs,
            descriptive_stats,
        )

        # Compute statistics per condition
        condition_stats = {}
        msci_by_condition = {}

        for condition, results in self.results_by_condition.items():
            successful = [r for r in results if r.success]
            msci_scores = [r.msci for r in successful if r.msci is not None]

            if msci_scores:
                msci_by_condition[condition] = msci_scores
                condition_stats[condition] = {
                    "n_total": len(results),
                    "n_successful": len(successful),
                    "success_rate": len(successful) / len(results),
                    "msci": descriptive_stats(msci_scores),
                    "mean_time_ms": np.mean([r.generation_time_ms for r in successful]),
                }

                # Token usage stats
                token_results = [r for r in successful if r.planner_metrics]
                if token_results:
                    total_tokens = [r.planner_metrics["total_tokens"] for r in token_results]
                    condition_stats[condition]["mean_tokens"] = np.mean(total_tokens)

        # Statistical comparisons
        statistical_tests = {}

        if len(msci_by_condition) >= 2:
            # Find minimum length for paired comparisons
            min_len = min(len(v) for v in msci_by_condition.values())

            if min_len >= 2:
                # Truncate to same length for paired tests
                truncated = {k: v[:min_len] for k, v in msci_by_condition.items()}
                comparisons = compare_all_pairs(truncated, paired=True)

                for key, result in comparisons.items():
                    statistical_tests[key] = result.to_dict()

        # Ablation-specific analysis
        ablation_analysis = self._analyze_ablation(msci_by_condition, condition_stats)

        return {
            "config": {
                "name": self.config.name,
                "conditions": self.config.conditions,
                "n_prompts": self.config.n_prompts,
                "n_seeds": self.config.n_seeds,
                "base_seed": self.config.base_seed,
            },
            "timestamp": timestamp,
            "n_prompts": len(prompts),
            "total_runs": len(self.results),
            "successful_runs": sum(1 for r in self.results if r.success),
            "condition_statistics": condition_stats,
            "statistical_tests": statistical_tests,
            "ablation_analysis": ablation_analysis,
            "raw_results": [r.to_dict() for r in self.results],
        }

    def _analyze_ablation(
        self,
        msci_by_condition: Dict[str, List[float]],
        condition_stats: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """
        Perform ablation-specific analysis.

        Key questions:
        1. Does single_planner improve over direct?
        2. Does council improve over single_planner?
        3. Does extended_prompt match council? (controls for tokens)
        """
        import numpy as np

        analysis = {
            "research_questions": {},
            "conclusions": [],
        }

        # RQ: Planning Effect (single_planner vs direct)
        if "direct" in msci_by_condition and "single_planner" in msci_by_condition:
            direct_mean = np.mean(msci_by_condition["direct"])
            single_mean = np.mean(msci_by_condition["single_planner"])
            diff = single_mean - direct_mean

            analysis["research_questions"]["planning_effect"] = {
                "comparison": "single_planner vs direct",
                "direct_mean": direct_mean,
                "single_planner_mean": single_mean,
                "difference": diff,
                "interpretation": "Planning improves MSCI" if diff > 0 else "No planning benefit",
            }

        # RQ: Council Structure (council vs single_planner)
        if "single_planner" in msci_by_condition and "council" in msci_by_condition:
            single_mean = np.mean(msci_by_condition["single_planner"])
            council_mean = np.mean(msci_by_condition["council"])
            diff = council_mean - single_mean

            analysis["research_questions"]["council_structure"] = {
                "comparison": "council vs single_planner",
                "single_planner_mean": single_mean,
                "council_mean": council_mean,
                "difference": diff,
                "interpretation": "Multi-agent structure helps" if diff > 0 else "No structural benefit",
            }

        # RQ: Token Control (extended_prompt vs council)
        if "extended_prompt" in msci_by_condition and "council" in msci_by_condition:
            extended_mean = np.mean(msci_by_condition["extended_prompt"])
            council_mean = np.mean(msci_by_condition["council"])
            diff = council_mean - extended_mean

            analysis["research_questions"]["token_control"] = {
                "comparison": "council vs extended_prompt (same token budget)",
                "extended_prompt_mean": extended_mean,
                "council_mean": council_mean,
                "difference": diff,
                "interpretation": (
                    "Council benefit is from STRUCTURE (not just more tokens)"
                    if diff > 0.01 else
                    "Council benefit is from TOKENS (not structure)"
                    if diff < -0.01 else
                    "Council and extended_prompt are equivalent"
                ),
            }

        # Overall conclusion
        if "token_control" in analysis["research_questions"]:
            tc = analysis["research_questions"]["token_control"]
            if tc["difference"] > 0.01:
                analysis["conclusions"].append(
                    "Council-Lite's benefit comes from its multi-agent STRUCTURE, "
                    "not just the increased token budget."
                )
            elif tc["difference"] < -0.01:
                analysis["conclusions"].append(
                    "Council-Lite's benefit is primarily from using more TOKENS. "
                    "The multi-agent structure provides no additional benefit."
                )
            else:
                analysis["conclusions"].append(
                    "Council-Lite and extended single prompting produce equivalent results. "
                    "The benefit is likely from increased compute/tokens."
                )

        return analysis

    def _print_summary(self, report: Dict[str, Any]):
        """Print formatted summary."""
        print(f"\n{'=' * 70}")
        print("ABLATION STUDY SUMMARY")
        print(f"{'=' * 70}")

        stats = report.get("condition_statistics", {})
        print("\nConditions ranked by mean MSCI:")
        sorted_stats = sorted(stats.items(), key=lambda x: x[1].get("msci", {}).get("mean", 0), reverse=True)
        for i, (cond, s) in enumerate(sorted_stats, 1):
            msci = s.get("msci", {})
            tokens = s.get("mean_tokens", "N/A")
            print(f"  {i}. {cond}: MSCI={msci.get('mean', 0):.4f}±{msci.get('std', 0):.4f}, tokens={tokens}")

        # Print ablation conclusions
        ablation = report.get("ablation_analysis", {})
        conclusions = ablation.get("conclusions", [])
        if conclusions:
            print("\n--- KEY FINDINGS ---")
            for conclusion in conclusions:
                print(f"  • {conclusion}")

        print(f"\n{'=' * 70}")
