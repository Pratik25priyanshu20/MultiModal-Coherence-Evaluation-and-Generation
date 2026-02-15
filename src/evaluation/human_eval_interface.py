"""
Human Evaluation CLI Interface

A simple command-line interface for collecting human coherence judgments.
Designed for single-rater evaluation with blind evaluation and bias mitigation.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from src.evaluation.human_eval_schema import (
    CoherenceRubric,
    EvaluationSample,
    EvaluationSession,
    HumanEvaluation,
)


class HumanEvalInterface:
    """
    CLI interface for human evaluation of multimodal coherence.

    Features:
    - Blind evaluation (condition labels hidden)
    - Randomized sample order
    - Session save/resume
    - Structured rubric display
    - Progress tracking
    """

    def __init__(
        self,
        samples: List[EvaluationSample],
        evaluator_id: str = "default",
        session_dir: Path = Path("evaluation/human_eval_sessions"),
        shuffle: bool = True,
        rerating_fraction: float = 0.20,
    ):
        """
        Initialize the evaluation interface.

        Args:
            samples: List of samples to evaluate
            evaluator_id: Identifier for the evaluator
            session_dir: Directory to save session state
            shuffle: Whether to randomize sample order (for blind evaluation)
            rerating_fraction: Fraction of samples to re-rate for reliability
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.rubric = CoherenceRubric()

        # Create or load session
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Shuffle for blind evaluation
        if shuffle:
            samples = samples.copy()
            random.shuffle(samples)

        # Select samples for re-rating (at the end)
        n_rerating = max(1, int(len(samples) * rerating_fraction))
        rerating_indices = random.sample(range(len(samples)), n_rerating)
        rerating_sample_ids = [samples[i].sample_id for i in rerating_indices]

        # Append rerating samples at the end
        rerating_samples = [samples[i] for i in rerating_indices]
        all_samples = samples + rerating_samples

        self.session = EvaluationSession(
            session_id=session_id,
            evaluator_id=evaluator_id,
            samples=all_samples,
            rerating_sample_ids=rerating_sample_ids,
        )

    @classmethod
    def resume_session(cls, session_path: Path) -> "HumanEvalInterface":
        """Resume an interrupted evaluation session."""
        instance = cls.__new__(cls)
        instance.session_dir = session_path.parent
        instance.session = EvaluationSession.load(session_path)
        instance.rubric = CoherenceRubric()
        return instance

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')

    def display_header(self):
        """Display session header with progress."""
        print("=" * 70)
        print("MULTIMODAL COHERENCE EVALUATION")
        print("=" * 70)
        print(f"Session: {self.session.session_id}")
        print(f"Evaluator: {self.session.evaluator_id}")
        print(f"Progress: {len(self.session.evaluations)}/{len(self.session.samples)} "
              f"({self.session.progress:.1f}%)")

        # Check if in rerating phase
        current = self.session.get_current_sample()
        if current and current.sample_id in self.session.rerating_sample_ids:
            n_original = len(self.session.samples) - len(self.session.rerating_sample_ids)
            if self.session.current_index >= n_original:
                print("\n[CONSISTENCY CHECK PHASE - Please rate as if first time]")

        print("=" * 70)

    def display_sample(self, sample: EvaluationSample):
        """Display sample content for evaluation (blind - no condition info)."""
        print(f"\n--- Sample {self.session.current_index + 1} ---\n")

        # Display text content
        print("TEXT CONTENT:")
        print("-" * 40)
        print(sample.text_content[:500] + ("..." if len(sample.text_content) > 500 else ""))
        print("-" * 40)

        # Display image path (user can open manually or we can try to open)
        print(f"\nIMAGE: {sample.image_path}")
        print("  (Press 'i' to open image in viewer)")

        # Display audio path
        print(f"\nAUDIO: {sample.audio_path}")
        print("  (Press 'a' to play audio)")

    def open_image(self, path: str):
        """Open image in system viewer."""
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", path], check=True)
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", path], check=True)
            elif sys.platform == "win32":
                os.startfile(path)
            print("  [Image opened in viewer]")
        except Exception as e:
            print(f"  [Could not open image: {e}]")

    def play_audio(self, path: str):
        """Play audio file."""
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", path], check=True)
            elif sys.platform == "linux":
                # Try common audio players
                for player in ["aplay", "paplay", "mpv", "ffplay"]:
                    try:
                        subprocess.run([player, path], check=True)
                        break
                    except FileNotFoundError:
                        continue
            elif sys.platform == "win32":
                os.startfile(path)
            print("  [Audio played]")
        except Exception as e:
            print(f"  [Could not play audio: {e}]")

    def display_rubric(self, dimension: str):
        """Display the rubric for a specific dimension."""
        rubrics = {
            "text_image": self.rubric.text_image_rubric,
            "text_audio": self.rubric.text_audio_rubric,
            "image_audio": self.rubric.image_audio_rubric,
            "overall": self.rubric.overall_rubric,
        }

        if dimension in rubrics:
            print(f"\n{dimension.upper().replace('_', '-')} COHERENCE RUBRIC:")
            for score, description in rubrics[dimension].items():
                print(f"  {score}: {description}")

    def get_rating(self, prompt: str, dimension: str) -> int:
        """Get a single rating with validation."""
        self.display_rubric(dimension)

        while True:
            try:
                user_input = input(f"\n{prompt} (1-5, 'r' for rubric, 'q' to quit): ").strip().lower()

                if user_input == 'q':
                    raise KeyboardInterrupt
                if user_input == 'r':
                    self.display_rubric(dimension)
                    continue

                rating = int(user_input)
                if 1 <= rating <= 5:
                    return rating
                print("  Please enter a number between 1 and 5.")
            except ValueError:
                print("  Invalid input. Please enter a number 1-5.")

    def collect_evaluation(self, sample: EvaluationSample) -> HumanEvaluation:
        """Collect all ratings for a single sample."""
        print("\n" + "=" * 50)
        print("RATE THE COHERENCE OF THIS MULTIMODAL CONTENT")
        print("=" * 50)

        # Check if this is a rerating
        is_rerating = (
            sample.sample_id in self.session.rerating_sample_ids and
            self.session.current_index >= len(self.session.samples) - len(self.session.rerating_sample_ids)
        )

        # Collect ratings
        text_image = self.get_rating(
            "Text-Image coherence", "text_image"
        )
        text_audio = self.get_rating(
            "Text-Audio coherence", "text_audio"
        )
        image_audio = self.get_rating(
            "Image-Audio coherence", "image_audio"
        )
        overall = self.get_rating(
            "Overall multimodal coherence", "overall"
        )

        # Confidence rating
        print("\nHow confident are you in these ratings?")
        confidence = self.get_rating(
            "Confidence (1=very uncertain, 5=very confident)", "confidence"
        )

        # Optional notes
        notes = input("\nAny notes or observations? (press Enter to skip): ").strip()

        return HumanEvaluation(
            sample_id=sample.sample_id,
            evaluator_id=self.session.evaluator_id,
            text_image_coherence=text_image,
            text_audio_coherence=text_audio,
            image_audio_coherence=image_audio,
            overall_coherence=overall,
            confidence=confidence,
            notes=notes,
            is_rerating=is_rerating,
        )

    def run_interactive_loop(self) -> bool:
        """
        Run the interactive evaluation loop.

        Returns:
            True if session completed, False if interrupted
        """
        print("\nStarting evaluation session...")
        print("Commands: 'i'=view image, 'a'=play audio, 'q'=quit (saves progress)")
        input("Press Enter to begin...")

        try:
            while not self.session.is_complete:
                self.clear_screen()
                self.display_header()

                sample = self.session.get_current_sample()
                if not sample:
                    break

                self.display_sample(sample)

                # Handle media viewing commands
                while True:
                    cmd = input("\nPress Enter to rate, 'i'=image, 'a'=audio, 'q'=quit: ").strip().lower()

                    if cmd == 'i':
                        self.open_image(sample.image_path)
                    elif cmd == 'a':
                        self.play_audio(sample.audio_path)
                    elif cmd == 'q':
                        raise KeyboardInterrupt
                    elif cmd == '':
                        break

                # Collect evaluation
                evaluation = self.collect_evaluation(sample)
                self.session.add_evaluation(evaluation)

                # Save after each evaluation
                self._save_session()

                print(f"\n✓ Evaluation saved ({self.session.progress:.1f}% complete)")
                input("Press Enter to continue...")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Progress saved.")
            self._save_session()
            return False

        print("\n" + "=" * 70)
        print("SESSION COMPLETE!")
        print("=" * 70)
        print(f"Total evaluations: {len(self.session.evaluations)}")
        self._save_session()

        return True

    def _save_session(self):
        """Save current session state."""
        session_path = self.session_dir / f"session_{self.session.session_id}.json"
        self.session.save(session_path)

    def get_session_path(self) -> Path:
        """Get the path to the current session file."""
        return self.session_dir / f"session_{self.session.session_id}.json"


def load_samples_from_runs(
    runs_dir: Path,
    n_samples: int = 100,
    conditions: Optional[List[str]] = None,
) -> List[EvaluationSample]:
    """
    Load samples from experiment runs for human evaluation.

    Args:
        runs_dir: Directory containing run bundles
        n_samples: Number of samples to load
        conditions: Optional list of conditions to filter

    Returns:
        List of EvaluationSample objects
    """
    samples = []
    runs_dir = Path(runs_dir)

    for bundle_path in sorted(runs_dir.glob("*/bundle.json")):
        try:
            with bundle_path.open("r", encoding="utf-8") as f:
                bundle = json.load(f)

            run_id = bundle_path.parent.name
            condition = bundle.get("meta", {}).get("condition", "unknown")
            mode = bundle.get("meta", {}).get("mode", "unknown")

            # Filter by condition if specified
            if conditions and f"{mode}_{condition}" not in conditions:
                continue

            # Get paths
            image_path = str(bundle_path.parent / "image" / "output.png")
            audio_path = str(bundle_path.parent / "audio" / "output.wav")

            # Check files exist
            if not Path(image_path).exists() or not Path(audio_path).exists():
                continue

            sample = EvaluationSample(
                sample_id=str(uuid.uuid4())[:8],
                text_content=bundle.get("outputs", {}).get("text", ""),
                image_path=image_path,
                audio_path=audio_path,
                condition=f"{mode}_{condition}",
                mode=mode,
                perturbation=condition,
                msci_score=bundle.get("scores", {}).get("msci"),
                run_id=run_id,
                original_prompt=bundle.get("prompts", {}).get("input", ""),
            )
            samples.append(sample)

            if len(samples) >= n_samples:
                break

        except Exception as e:
            print(f"Warning: Could not load {bundle_path}: {e}")
            continue

    return samples


def create_balanced_sample_set(
    runs_dir: Path,
    samples_per_condition: int = 17,
    conditions: Optional[List[str]] = None,
) -> List[EvaluationSample]:
    """
    Create a balanced sample set with equal representation across conditions.

    For 6 conditions × 17 samples = 102 samples (close to target 100)

    Args:
        runs_dir: Directory containing run bundles
        samples_per_condition: Number of samples per condition
        conditions: List of conditions to include

    Returns:
        Balanced list of EvaluationSample objects
    """
    if conditions is None:
        conditions = [
            "direct_baseline",
            "direct_wrong_image",
            "direct_wrong_audio",
            "planner_baseline",
            "planner_wrong_image",
            "planner_wrong_audio",
        ]

    all_samples = []

    for condition in conditions:
        mode, perturbation = condition.rsplit("_", 1)
        condition_samples = load_samples_from_runs(
            runs_dir=runs_dir,
            n_samples=samples_per_condition,
            conditions=[condition],
        )
        all_samples.extend(condition_samples[:samples_per_condition])
        print(f"Loaded {len(condition_samples[:samples_per_condition])} samples for {condition}")

    print(f"Total samples: {len(all_samples)}")
    return all_samples
