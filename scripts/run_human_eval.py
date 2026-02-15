#!/usr/bin/env python3
"""
RQ3 Human Evaluation Runner — Multi-rater blind evaluation.

All evaluators rate the same 30 samples (+ 6 re-ratings for intra-rater
reliability). Samples are presented in randomized order with condition
labels hidden.

Usage:
    python scripts/run_human_eval.py --evaluator alice       # new session
    python scripts/run_human_eval.py --evaluator alice --resume  # resume
    python scripts/run_human_eval.py --list                  # list sessions

Prerequisite:
    python scripts/select_rq3_samples.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.human_eval_schema import EvaluationSample
from src.evaluation.human_eval_interface import HumanEvalInterface

SAMPLES_PATH = PROJECT_ROOT / "runs" / "rq3" / "rq3_samples.json"
SESSION_DIR = PROJECT_ROOT / "runs" / "rq3" / "sessions"


def load_rq3_samples() -> list:
    """Load the fixed 30-sample set from RQ3 selection."""
    if not SAMPLES_PATH.exists():
        print(f"ERROR: {SAMPLES_PATH} not found.")
        print("Run first:  python scripts/select_rq3_samples.py")
        sys.exit(1)

    with open(SAMPLES_PATH) as f:
        data = json.load(f)

    samples = []
    for s in data["samples"]:
        samples.append(EvaluationSample(
            sample_id=s["sample_id"],
            text_content=s["prompt_text"],
            image_path=s["image_path"],
            audio_path=s["audio_path"],
            condition=s["condition"],
            mode=s.get("mode", "direct"),
            perturbation=s["condition"],
            msci_score=s["msci"],
            run_id=f"{s['source']}_{s['prompt_id']}_s{s['seed']}",
            original_prompt=s["prompt_text"],
        ))

    return samples


def find_session(evaluator_id: str) -> Path | None:
    """Find existing session file for an evaluator."""
    if not SESSION_DIR.exists():
        return None
    for p in sorted(SESSION_DIR.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            if data.get("evaluator_id") == evaluator_id:
                return p
        except Exception:
            continue
    return None


def list_sessions():
    """List all evaluation sessions with progress."""
    if not SESSION_DIR.exists():
        print("No sessions found.")
        return

    print("\n" + "=" * 70)
    print("RQ3 EVALUATION SESSIONS")
    print("=" * 70)

    sessions = sorted(SESSION_DIR.glob("*.json"))
    if not sessions:
        print("No sessions found.")
        return

    for p in sessions:
        try:
            with open(p) as f:
                data = json.load(f)
            n_evals = len(data.get("evaluations", []))
            n_samples = len(data.get("samples", []))
            pct = n_evals / n_samples * 100 if n_samples else 0
            status = "COMPLETE" if data.get("completed_at") else "IN PROGRESS"
            print(f"  {data['evaluator_id']:15s}  {n_evals}/{n_samples} ({pct:.0f}%)  [{status}]")
        except Exception as e:
            print(f"  [ERROR] {p.name}: {e}")

    print()


def main():
    parser = argparse.ArgumentParser(description="RQ3 Human Evaluation")
    parser.add_argument("--evaluator", "-e", type=str, help="Evaluator name/ID")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume existing session")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all sessions")
    args = parser.parse_args()

    if args.list:
        list_sessions()
        return

    if not args.evaluator:
        parser.error("--evaluator is required (or use --list)")

    evaluator_id = args.evaluator.strip().lower()

    # Resume mode
    if args.resume:
        session_path = find_session(evaluator_id)
        if not session_path:
            print(f"No existing session for '{evaluator_id}'.")
            print("Start new session by running without --resume.")
            return

        print(f"Resuming session for '{evaluator_id}'")
        interface = HumanEvalInterface.resume_session(session_path)

    # New session
    else:
        existing = find_session(evaluator_id)
        if existing:
            print(f"Session already exists for '{evaluator_id}'.")
            print("Use --resume to continue, or choose a different evaluator ID.")
            return

        samples = load_rq3_samples()

        interface = HumanEvalInterface(
            samples=samples,
            evaluator_id=evaluator_id,
            session_dir=SESSION_DIR,
            shuffle=True,
            rerating_fraction=0.20,
        )

    # Display session info
    n_original = len(interface.session.samples) - len(interface.session.rerating_sample_ids)
    n_rerating = len(interface.session.rerating_sample_ids)

    print("\n" + "=" * 60)
    print("RQ3 HUMAN EVALUATION")
    print("=" * 60)
    print(f"Evaluator:   {interface.session.evaluator_id}")
    print(f"Samples:     {n_original} + {n_rerating} re-ratings = {len(interface.session.samples)}")
    print(f"Progress:    {interface.session.progress:.1f}%")
    print(f"Est. time:   ~20-30 minutes")
    print("=" * 60)
    print("\nFor each sample you will see text, an image, and audio.")
    print("Rate the coherence between modalities on a 1-5 scale.")
    print("Press 'i' to view image, 'a' to play audio.")

    # Run evaluation
    completed = interface.run_interactive_loop()

    if completed:
        print(f"\nSession complete! Saved to: {SESSION_DIR}")
        print(f"\nOnce all evaluators finish:")
        print(f"  python scripts/analyze_rq3.py")
    else:
        print(f"\nSession saved. Resume with:")
        print(f"  python scripts/run_human_eval.py -e {evaluator_id} --resume")


if __name__ == "__main__":
    main()
