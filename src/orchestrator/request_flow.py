from __future__ import annotations

# Phase-3C: Targeted Modality Retry Controller

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.coherence.drift_detector import detect_drift
from src.coherence.msci import compute_msci_v0
from src.coherence.reporting import build_final_assessment
from src.coherence.scorer import CoherenceScorer
from src.coherence.controller import route_retry
from src.coherence.retry.retry_si_a import retry_si_a
from src.coherence.retry.retry_st_i import retry_st_i
from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.generators.audio.generator import AudioGenerator
from src.generators.image.generator import ImageRetrievalGenerator
from src.generators.text.generator import TextGenerator
from src.narrative.generator import NarrativeGenerator
from src.orchestrator.regeneration_policy import decide_regeneration
from src.orchestrator.run_manager import create_run_paths
from src.planner.council import SemanticPlanningCouncil
from src.planner.schema import SemanticPlan
from src.planner.schema_to_text import plan_to_canonical_text
from src.storage.metadata import write_run_metadata


@dataclass(frozen=True)
class RunOutput:
    run_id: str
    semantic_plan: Dict[str, Any]
    merge_report: Dict[str, Any]
    planner_outputs: Dict[str, Any]
    narrative_structured: Dict[str, Any]
    narrative_text: str
    image_path: str
    audio_path: str
    scores: Dict[str, Any]
    coherence: Dict[str, Any]
    final_assessment: Dict[str, Any]
    drift: Dict[str, bool]
    attempts: int
    decisions: List[Dict[str, Any]]


class Orchestrator:
    def __init__(
        self,
        council: SemanticPlanningCouncil,
        text_gen: TextGenerator,
        image_gen: ImageRetrievalGenerator,
        audio_gen: AudioGenerator,
        msci_threshold: float = 0.42,
        max_attempts: int = 4,
        runs_dir: str = "runs",
    ):
        self.council = council
        self.text_gen = text_gen
        self.image_gen = image_gen
        self.audio_gen = audio_gen
        self.msci_threshold = msci_threshold
        self.max_attempts = max_attempts
        self.runs_dir = runs_dir

        self.embedder = AlignedEmbedder(target_dim=512)
        self.narrative_generator = NarrativeGenerator()
        self.coherence_scorer = CoherenceScorer()

    def run(self, user_prompt: str) -> RunOutput:
        paths = create_run_paths(self.runs_dir)

        council_result = self.council.run(user_prompt)
        if isinstance(council_result, SemanticPlan):
            plan = council_result
            merge_report = {
                "agreement_score": 1.0,
                "per_section_agreement": {},
                "conflicts": {},
                "notes": "unified_planner",
            }
            planner_outputs = {"unified": plan.model_dump()}
        else:
            plan = council_result.merged_plan
            merge_report = {
                "agreement_score": council_result.merge_report.agreement_score,
                "per_section_agreement": council_result.merge_report.per_section_agreement,
                "conflicts": council_result.merge_report.conflicts,
                "notes": council_result.merge_report.notes,
            }
            planner_outputs = {
                "plan_a": council_result.plan_a.model_dump(),
                "plan_b": council_result.plan_b.model_dump(),
                "plan_c": council_result.plan_c.model_dump(),
            }
        plan_text = plan_to_canonical_text(plan)

        plan_embedding = self.embedder.embed_text(plan_text)

        img_pool = self.image_gen.retrieve_top_k(plan_text, k=8)
        if not img_pool:
            index_path = getattr(self.image_gen, "index_path", None)
            hint = f" Expected index at {index_path}." if index_path else ""
            raise RuntimeError(
                "No image candidates retrieved. Build the image index or switch to a"
                f" generative image backend.{hint}"
            )

        best_state: Optional[
            Tuple[float, str, str, str, Dict[str, Any], Dict[str, bool], int]
        ] = None
        decisions: List[Dict[str, Any]] = []
        retry_outcomes: List[Dict[str, Any]] = []

        narrative_structured = self.narrative_generator.generate(plan.model_dump())
        narrative = narrative_structured.combined_scene

        image_path = img_pool[0][0]
        audio_path = str(paths.audio_dir / "audio_attempt1.wav")

        audio_prompt = (
            f"{plan.scene_summary}. Soundscape: {', '.join(plan.audio_elements)}. "
            f"Mood: {', '.join(plan.mood_emotion)}."
        )
        retry_analysis: List[Dict[str, Any]] = []

        epsilon = 0.01
        for attempt in range(1, self.max_attempts + 1):
            if attempt == 1:
                audio_result = self.audio_gen.generate(audio_prompt, audio_path)
                audio_path = audio_result.audio_path
                audio_backend = audio_result.backend
            else:
                last_scores = decisions[-1]["scores"]
                last_coherence = decisions[-1].get("coherence", {})
                classification = last_coherence.get("classification", {})
                context = {
                    "semantic_plan": plan.model_dump(),
                    "narrative_structured": narrative_structured.model_dump(),
                    "plan_text": plan_text,
                    "image_path": image_path,
                    "audio_path": audio_path,
                    "image_generator": self.image_gen,
                    "audio_generator": self.audio_gen,
                }

                retry_action = None
                retry_strategy = None
                retry_metric = None
                retry_trigger = classification.get("label")
                handled_regen = False

                if (
                    classification.get("label") == "MODALITY_FAILURE"
                    and classification.get("weakest_metric") == "st_i"
                ):
                    context = retry_st_i(context)
                    image_path = context.get("image") or context.get("image_path") or image_path
                    retry_strategy = "ALIGN_IMAGE_TO_TEXT"
                    retry_metric = "st_i"
                    retry_action = {
                        "regenerate": "image",
                        "failed_metric": "st_i",
                        "strategy": retry_strategy,
                    }
                    handled_regen = True
                elif (
                    classification.get("label") == "MODALITY_FAILURE"
                    and classification.get("weakest_metric") == "si_a"
                ):
                    audio_retry_path = str(paths.audio_dir / f"audio_attempt{attempt}.wav")
                    context["audio_path"] = audio_retry_path
                    context = retry_si_a(context)
                    audio_path = context.get("audio") or context.get("audio_path") or audio_path
                    audio_backend = context.get("audio_backend")
                    retry_meta = context.get("retry", {})
                    retry_strategy = retry_meta.get("strategy", "ALIGN_AUDIO_TO_IMAGE")
                    retry_metric = "si_a"
                    retry_action = {
                        "regenerate": "audio",
                        "failed_metric": "si_a",
                        "strategy": retry_strategy,
                    }
                    handled_regen = True
                else:
                    retry_action = route_retry(classification, context)

                if retry_action and retry_action.get("regenerate") == "full":
                    retry_strategy = retry_action.get("strategy")
                    retry_metric = retry_action.get("failed_metric")
                    handled_regen = True

                    council_result = self.council.run(user_prompt)
                    if isinstance(council_result, SemanticPlan):
                        plan = council_result
                        merge_report = {
                            "agreement_score": 1.0,
                            "per_section_agreement": {},
                            "conflicts": {},
                            "notes": "unified_planner",
                        }
                        planner_outputs = {"unified": plan.model_dump()}
                    else:
                        plan = council_result.merged_plan
                        merge_report = {
                            "agreement_score": council_result.merge_report.agreement_score,
                            "per_section_agreement": council_result.merge_report.per_section_agreement,
                            "conflicts": council_result.merge_report.conflicts,
                            "notes": council_result.merge_report.notes,
                        }
                        planner_outputs = {
                            "plan_a": council_result.plan_a.model_dump(),
                            "plan_b": council_result.plan_b.model_dump(),
                            "plan_c": council_result.plan_c.model_dump(),
                        }

                    plan_text = plan_to_canonical_text(plan)
                    plan_embedding = self.embedder.embed_text(plan_text)
                    narrative_structured = self.narrative_generator.generate(plan.model_dump())
                    narrative = narrative_structured.combined_scene

                    img_pool = self.image_gen.retrieve_top_k(plan_text, k=8)
                    if not img_pool:
                        index_path = getattr(self.image_gen, "index_path", None)
                        hint = f" Expected index at {index_path}." if index_path else ""
                        raise RuntimeError(
                            "No image candidates retrieved. Build the image index or switch to a"
                            f" generative image backend.{hint}"
                        )
                    image_path = img_pool[0][0]

                    audio_prompt = (
                        f"{plan.scene_summary}. Soundscape: {', '.join(plan.audio_elements)}. "
                        f"Mood: {', '.join(plan.mood_emotion)}."
                    )
                    audio_path = str(paths.audio_dir / f"audio_attempt{attempt}.wav")
                    audio_result = self.audio_gen.generate(audio_prompt, audio_path)
                    audio_path = audio_result.audio_path
                    audio_backend = audio_result.backend
                    target = "full"
                elif retry_action and retry_action.get("regenerate") in {"audio", "image"}:
                    target = retry_action["regenerate"]
                    retry_strategy = retry_action.get("strategy")
                    retry_metric = retry_action.get("failed_metric")
                    if target == "audio" and retry_action.get("audio_prompt"):
                        audio_prompt = retry_action["audio_prompt"]
                    if target == "image" and retry_action.get("image_prompt"):
                        img_pool = self.image_gen.retrieve_top_k(
                            retry_action["image_prompt"],
                            k=8,
                        )
                else:
                    target = decide_regeneration(
                        last_scores["msci"],
                        last_scores["st_i"],
                        last_scores["st_a"],
                        self.msci_threshold,
                    )

                if not handled_regen and target == "image":
                    idx = min(attempt - 1, max(len(img_pool) - 1, 0))
                    image_path = img_pool[idx][0] if img_pool else image_path
                elif not handled_regen and target == "audio":
                    audio_path = str(paths.audio_dir / f"audio_attempt{attempt}.wav")
                    audio_prompt_variant = audio_prompt + f" Intensity level: {attempt}."
                    audio_result = self.audio_gen.generate(audio_prompt_variant, audio_path)
                    audio_backend = audio_result.backend
                elif not handled_regen and target == "text":
                    narrative = self.text_gen.generate(
                        f"{plan_text}\n\nRewrite concisely, keep the same meaning:\n"
                    ).text
                else:
                    target = "none"

            if not image_path:
                raise RuntimeError("Image path is empty; retrieval produced no candidates.")
            image_emb = self.embedder.embed_image(image_path)
            audio_emb = self.embedder.embed_audio(audio_path)

            msci = compute_msci_v0(
                plan_embedding,
                image_emb,
                audio_emb,
                include_image_audio=True,
            )
            drift = detect_drift(msci.msci, msci.st_i, msci.st_a, msci.si_a)

            scores = {
                "msci": msci.msci,
                "st_i": msci.st_i,
                "st_a": msci.st_a,
                "si_a": msci.si_a,
                "agreement_score": merge_report["agreement_score"],
                "per_section_agreement": merge_report["per_section_agreement"],
            }
            metric_scores = {k: scores[k] for k in ("msci", "st_i", "st_a", "si_a")}
            coherence_step = self.coherence_scorer.score(
                scores=metric_scores,
                global_drift=drift["global_drift"],
            )
            coherence_step["needs_repair"] = (
                coherence_step["classification"]["label"] == "MODALITY_FAILURE"
                and coherence_step["classification"]["weakest_metric"] == "st_i"
            )

            repair_attempts = 0
            while coherence_step["needs_repair"] and repair_attempts < 2:
                narrative_structured = self.narrative_generator.repair_visual_description(
                    plan.model_dump(),
                    image_path=image_path,
                )
                narrative = narrative_structured.combined_scene
                plan_embedding = self.embedder.embed_text(
                    narrative_structured.visual_description
                )

                msci = compute_msci_v0(
                    plan_embedding,
                    image_emb,
                    audio_emb,
                    include_image_audio=True,
                )
                drift = detect_drift(msci.msci, msci.st_i, msci.st_a, msci.si_a)

                scores = {
                    "msci": msci.msci,
                    "st_i": msci.st_i,
                    "st_a": msci.st_a,
                    "si_a": msci.si_a,
                    "agreement_score": merge_report["agreement_score"],
                    "per_section_agreement": merge_report["per_section_agreement"],
                }
                metric_scores = {k: scores[k] for k in ("msci", "st_i", "st_a", "si_a")}
                coherence_step = self.coherence_scorer.score(
                    scores=metric_scores,
                    global_drift=drift["global_drift"],
                )
                coherence_step["needs_repair"] = (
                    coherence_step["classification"]["label"] == "MODALITY_FAILURE"
                    and coherence_step["classification"]["weakest_metric"] == "st_i"
                )
                repair_attempts += 1

                if coherence_step["classification"]["label"] in {
                    "HIGH_COHERENCE",
                    "LOCAL_MODALITY_WEAKNESS",
                }:
                    break

            step_decision = {
                "attempt": attempt,
                "image_path": image_path,
                "audio_path": audio_path,
                "audio_backend": audio_backend if "audio_backend" in locals() else None,
                "scores": scores,
                "coherence": coherence_step,
                "drift": drift,
                "retry_strategy": retry_strategy if attempt > 1 else None,
                "retry_metric": retry_metric if attempt > 1 else None,
            }
            decisions.append(step_decision)

            if attempt > 1 and retry_metric:
                prev_scores = decisions[-2].get("scores", {})
                before = prev_scores.get(retry_metric)
                after = scores.get(retry_metric)
                if before is not None and after is not None:
                    before_status = self.coherence_scorer.thresholds.classify_value(
                        retry_metric,
                        before,
                    )
                    after_status = self.coherence_scorer.thresholds.classify_value(
                        retry_metric,
                        after,
                    )
                    success = (before_status == "FAIL" and after_status in {"WEAK", "GOOD"}) or (
                        after > before + epsilon
                    )
                    retry_outcomes.append(
                        {
                            "strategy": retry_strategy,
                            "trigger": retry_trigger,
                            "weakest_metric": retry_metric,
                            "before": {
                                "msci": prev_scores.get("msci"),
                                "st_i": prev_scores.get("st_i"),
                                "st_a": prev_scores.get("st_a"),
                                "si_a": prev_scores.get("si_a"),
                            },
                            "after": {
                                "msci": scores.get("msci"),
                                "st_i": scores.get("st_i"),
                                "st_a": scores.get("st_a"),
                                "si_a": scores.get("si_a"),
                            },
                            "epsilon": epsilon,
                            "success": success,
                        }
                    )

            if best_state is None or scores["msci"] > best_state[0]:
                best_state = (
                    scores["msci"],
                    narrative,
                    image_path,
                    audio_path,
                    scores,
                    drift,
                    attempt,
                )

            if scores["msci"] >= self.msci_threshold and not drift["global_drift"]:
                break

        assert best_state is not None
        _, best_text, best_img, best_aud, best_scores, best_drift, best_attempt = best_state

        metric_scores = {k: best_scores[k] for k in ("msci", "st_i", "st_a", "si_a") if k in best_scores}
        coherence = self.coherence_scorer.score(
            scores=metric_scores,
            global_drift=best_drift["global_drift"],
        )
        final_assessment = build_final_assessment(coherence, retry_outcomes)

        out = RunOutput(
            run_id=paths.run_id,
            semantic_plan=plan.model_dump(),
            merge_report=merge_report,
            planner_outputs=planner_outputs,
            narrative_structured=narrative_structured.model_dump(),
            narrative_text=best_text,
            image_path=best_img,
            audio_path=best_aud,
            scores=best_scores,
            coherence=coherence,
            final_assessment=final_assessment,
            drift=best_drift,
            attempts=best_attempt,
            decisions=decisions,
        )

        write_run_metadata(
            paths.logs_dir / "run.json",
            {
                "run_id": out.run_id,
                "user_prompt": user_prompt,
                "semantic_plan": out.semantic_plan,
                "merge_report": out.merge_report,
                "planner_outputs": out.planner_outputs,
                "narrative_structured": out.narrative_structured,
                "final": {
                    "narrative_text": out.narrative_text,
                    "image_path": out.image_path,
                    "audio_path": out.audio_path,
                    "scores": out.scores,
                    "coherence": out.coherence,
                    "final_assessment": out.final_assessment,
                    "drift": out.drift,
                    "attempts": out.attempts,
                },
                "attempt_history": out.decisions,
            },
        )
        if retry_outcomes:
            write_run_metadata(
                paths.logs_dir / "retry_outcome.json",
                {"retries": retry_outcomes},
            )

        return out
