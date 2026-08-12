"""
Streamlit Human Evaluation App for Multimodal Coherence (HF Spaces)

Self-contained version for deployment to Hugging Face Spaces.
No ML models or torch dependencies — just media serving + rating collection.

Launch locally:
    streamlit run deploy/hf-eval/app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Paths (self-contained — no src.config.settings dependency)
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent
SAMPLES_PATH = APP_ROOT / "samples" / "rq3_samples_extended.json"
SAMPLES_PATH_30 = APP_ROOT / "samples" / "rq3_samples.json"
SESSIONS_DIR = APP_ROOT / "sessions"
MEDIA_DIR = APP_ROOT / "media"

RERATING_FRACTION = 0.20

# ---------------------------------------------------------------------------
# Persistent storage via HuggingFace Dataset repo
# ---------------------------------------------------------------------------
HF_DATASET_REPO = "pratik-250620/multimodal-coherence-eval-data"
HF_TOKEN = os.environ.get("HF_TOKEN")

_hf_api = None

def _get_hf_api():
    """Lazy-init HfApi with token."""
    global _hf_api
    if _hf_api is None and HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            _hf_api = HfApi(token=HF_TOKEN)
        except Exception:
            pass
    return _hf_api


def _sync_to_hf(session_path: Path):
    """Upload a session JSON to the persistent HF dataset repo."""
    api = _get_hf_api()
    if api is None:
        return
    try:
        api.upload_file(
            path_or_fileobj=str(session_path),
            path_in_repo=f"sessions/{session_path.name}",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
        )
    except Exception as e:
        # Non-fatal — local copy still exists
        st.toast(f"Cloud sync warning: {e}", icon="\u26a0\ufe0f")


def _restore_from_hf():
    """On startup, pull any existing sessions from HF dataset repo."""
    api = _get_hf_api()
    if api is None:
        return
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        files = api.list_repo_files(HF_DATASET_REPO, repo_type="dataset")
        session_files = [f for f in files if f.startswith("sessions/") and f.endswith(".json")]
        for fname in session_files:
            local_path = SESSIONS_DIR / Path(fname).name
            if not local_path.exists():
                api.hf_hub_download(
                    repo_id=HF_DATASET_REPO,
                    repo_type="dataset",
                    filename=fname,
                    local_dir=APP_ROOT,
                )
    except Exception:
        pass  # Non-fatal — fresh start is fine


# Restore sessions from HF on first load
if "hf_restored" not in st.session_state:
    _restore_from_hf()
    st.session_state["hf_restored"] = True


# ===========================================================================
# Inlined schema (from src/evaluation/human_eval_schema.py)
# ===========================================================================

@dataclass
class CoherenceRubric:
    text_image_rubric: Dict[int, str] = field(default_factory=lambda: {
        1: "Completely unrelated: Image has no semantic connection to text",
        2: "Vague thematic connection only: General theme matches but specifics differ",
        3: "Partial match: Some elements align, others clearly don't",
        4: "Mostly aligned: Most elements match, minor discrepancies",
        5: "Strong semantic alignment: Image accurately represents text content",
    })
    text_audio_rubric: Dict[int, str] = field(default_factory=lambda: {
        1: "Completely unrelated: Audio has no connection to described scene",
        2: "Vague connection: General mood might match but sounds don't fit",
        3: "Partial match: Some sounds fit the scene, others are mismatched",
        4: "Mostly aligned: Audio largely fits the scene with minor issues",
        5: "Strong alignment: Audio perfectly complements the described scene",
    })
    image_audio_rubric: Dict[int, str] = field(default_factory=lambda: {
        1: "Completely unrelated: Audio doesn't match what's shown in image",
        2: "Vague connection: Mood might match but sounds don't fit visuals",
        3: "Partial match: Some sounds plausible for image, others not",
        4: "Mostly aligned: Audio largely fits the visual scene",
        5: "Strong alignment: Audio sounds exactly right for the visual",
    })
    overall_rubric: Dict[int, str] = field(default_factory=lambda: {
        1: "No coherence: Modalities feel randomly combined",
        2: "Weak coherence: Some connection but feels disjointed",
        3: "Moderate coherence: Works together with noticeable gaps",
        4: "Good coherence: Modalities complement each other well",
        5: "Excellent coherence: Unified, immersive multimodal experience",
    })


@dataclass
class HumanEvaluation:
    sample_id: str
    evaluator_id: str
    text_image_coherence: int
    text_audio_coherence: int
    image_audio_coherence: int
    overall_coherence: int
    confidence: int = 3
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: str = ""
    is_rerating: bool = False

    def __post_init__(self):
        for attr in ['text_image_coherence', 'text_audio_coherence',
                     'image_audio_coherence', 'overall_coherence', 'confidence']:
            value = getattr(self, attr)
            if not 1 <= value <= 5:
                raise ValueError(f"{attr} must be between 1 and 5, got {value}")

    def mean_pairwise_score(self) -> float:
        return (self.text_image_coherence + self.text_audio_coherence +
                self.image_audio_coherence) / 3.0

    def weighted_score(self, w_ti: float = 0.45, w_ta: float = 0.45,
                       w_ia: float = 0.10) -> float:
        total = w_ti + w_ta + w_ia
        return (w_ti * self.text_image_coherence +
                w_ta * self.text_audio_coherence +
                w_ia * self.image_audio_coherence) / (total * 5)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "evaluator_id": self.evaluator_id,
            "text_image_coherence": self.text_image_coherence,
            "text_audio_coherence": self.text_audio_coherence,
            "image_audio_coherence": self.image_audio_coherence,
            "overall_coherence": self.overall_coherence,
            "confidence": self.confidence,
            "notes": self.notes,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "is_rerating": self.is_rerating,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanEvaluation":
        return cls(**data)


@dataclass
class EvaluationSample:
    sample_id: str
    text_content: str
    image_path: str
    audio_path: str
    condition: str = ""
    mode: str = ""
    perturbation: str = ""
    msci_score: Optional[float] = None
    run_id: str = ""
    original_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text_content": self.text_content,
            "image_path": self.image_path,
            "audio_path": self.audio_path,
            "condition": self.condition,
            "mode": self.mode,
            "perturbation": self.perturbation,
            "msci_score": self.msci_score,
            "run_id": self.run_id,
            "original_prompt": self.original_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationSample":
        return cls(**data)


@dataclass
class EvaluationSession:
    session_id: str
    evaluator_id: str
    samples: List[EvaluationSample]
    evaluations: List[HumanEvaluation] = field(default_factory=list)
    current_index: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    rerating_sample_ids: List[str] = field(default_factory=list)

    @property
    def progress(self) -> float:
        if not self.samples:
            return 0.0
        return len(self.evaluations) / len(self.samples) * 100

    @property
    def is_complete(self) -> bool:
        return len(self.evaluations) >= len(self.samples)

    def get_current_sample(self) -> Optional[EvaluationSample]:
        if self.current_index < len(self.samples):
            return self.samples[self.current_index]
        return None

    def add_evaluation(self, evaluation: HumanEvaluation):
        evaluation.session_id = self.session_id
        self.evaluations.append(evaluation)
        self.current_index += 1
        if self.is_complete:
            self.completed_at = datetime.now().isoformat()

    def save(self, path: Path):
        data = {
            "session_id": self.session_id,
            "evaluator_id": self.evaluator_id,
            "samples": [s.to_dict() for s in self.samples],
            "evaluations": [e.to_dict() for e in self.evaluations],
            "current_index": self.current_index,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "rerating_sample_ids": self.rerating_sample_ids,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "EvaluationSession":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            session_id=data["session_id"],
            evaluator_id=data["evaluator_id"],
            samples=[EvaluationSample.from_dict(s) for s in data["samples"]],
            evaluations=[HumanEvaluation.from_dict(e) for e in data["evaluations"]],
            current_index=data["current_index"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            rerating_sample_ids=data.get("rerating_sample_ids", []),
        )


# ===========================================================================
# End inlined schema
# ===========================================================================

RUBRIC = CoherenceRubric()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_samples_path() -> Path:
    """Pick sample file based on session state selector (default: 100)."""
    use_30 = st.session_state.get("dataset_size") == "30 samples (original)"
    path = SAMPLES_PATH_30 if use_30 else SAMPLES_PATH
    if path.exists():
        return path
    # Fallback
    return SAMPLES_PATH if SAMPLES_PATH.exists() else SAMPLES_PATH_30


def load_rq3_samples() -> List[EvaluationSample]:
    """Load evaluation samples from JSON, resolving media paths."""
    path = _resolve_samples_path()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for s in data["samples"]:
        # Resolve relative media paths via MEDIA_DIR
        image_path = str(MEDIA_DIR / s["image_path"])
        audio_path = str(MEDIA_DIR / s["audio_path"])

        samples.append(EvaluationSample(
            sample_id=s["sample_id"],
            text_content=s["prompt_text"],
            image_path=image_path,
            audio_path=audio_path,
            condition=s.get("condition", ""),
            mode=s.get("mode", ""),
            perturbation=s.get("condition", ""),
            msci_score=s.get("msci"),
            run_id=s.get("prompt_id", ""),
            original_prompt=s.get("prompt_text", ""),
        ))
    return samples


def shuffled_with_reratings(
    samples: List[EvaluationSample],
    evaluator_name: str,
) -> tuple[List[EvaluationSample], List[str]]:
    """Shuffle samples deterministically per evaluator, append re-rating subset."""
    seed = int(hashlib.sha256(evaluator_name.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    ordered = samples.copy()
    rng.shuffle(ordered)

    n_rerate = max(1, int(len(ordered) * RERATING_FRACTION))
    rerate_indices = rng.sample(range(len(ordered)), n_rerate)
    rerate_ids = [ordered[i].sample_id for i in rerate_indices]

    all_samples = ordered + [ordered[i] for i in rerate_indices]
    return all_samples, rerate_ids


def create_new_session(evaluator_name: str) -> EvaluationSession:
    raw_samples = load_rq3_samples()
    all_samples, rerate_ids = shuffled_with_reratings(raw_samples, evaluator_name)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    session = EvaluationSession(
        session_id=session_id,
        evaluator_id=evaluator_name,
        samples=all_samples,
        rerating_sample_ids=rerate_ids,
    )
    return session


def save_session(session: EvaluationSession):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"session_{session.session_id}.json"
    session.save(path)
    _sync_to_hf(path)


def find_existing_sessions(evaluator_name: str) -> List[Path]:
    if not SESSIONS_DIR.exists():
        return []
    results = []
    for p in sorted(SESSIONS_DIR.glob("session_*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("evaluator_id") == evaluator_name:
                results.append(p)
        except Exception:
            continue
    return results


def list_all_sessions() -> List[dict]:
    if not SESSIONS_DIR.exists():
        return []
    summaries = []
    for p in sorted(SESSIONS_DIR.glob("session_*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            summaries.append({
                "evaluator": data["evaluator_id"],
                "session_id": data["session_id"],
                "progress": f"{len(data['evaluations'])}/{len(data['samples'])}",
                "completed": data.get("completed_at") is not None,
                "path": str(p),
            })
        except Exception:
            continue
    return summaries


def build_sessions_zip() -> bytes:
    """Create a ZIP archive of all session JSON files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if SESSIONS_DIR.exists():
            for p in sorted(SESSIONS_DIR.glob("session_*.json")):
                zf.write(p, p.name)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Streamlit pages
# ---------------------------------------------------------------------------

def page_login():
    """Login / session management page."""
    st.title("Multimodal Coherence Evaluation")
    st.markdown("Rate the coherence of text + image + audio bundles.")

    # --- Storage info ---
    if HF_TOKEN:
        st.success(
            "**Cloud storage enabled.** Your ratings are automatically saved "
            "and will persist across Space restarts."
        )
    else:
        st.warning(
            "**Ephemeral storage:** Session files may be lost if the Space restarts. "
            "**Please download your ratings JSON after completing your session.**"
        )

    # --- Dataset (fixed to extended 100-sample set) ---
    if not SAMPLES_PATH.exists():
        st.error("Sample file not found. Check the `samples/` directory.")
        return
    st.session_state["dataset_size"] = "100 samples (extended)"

    evaluator_name = st.text_input("Your name", key="login_name")

    if not evaluator_name:
        st.info("Enter your name to begin.")
        _show_admin_panel()
        return

    existing = find_existing_sessions(evaluator_name)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start New Session", use_container_width=True):
            session = create_new_session(evaluator_name)
            save_session(session)
            st.session_state["session"] = session
            st.session_state["page"] = "eval"
            st.rerun()

    with col2:
        if existing:
            if st.button("Resume Session", use_container_width=True):
                session = EvaluationSession.load(existing[-1])
                st.session_state["session"] = session
                st.session_state["page"] = "eval"
                st.rerun()
        else:
            st.button("Resume Session", disabled=True, use_container_width=True,
                       help="No existing session found for this name")

    if existing:
        st.caption(f"Found {len(existing)} existing session(s) for **{evaluator_name}**.")

    _show_admin_panel()


def _show_admin_panel():
    """Admin overview with session list and ZIP download."""
    with st.expander("Admin panel"):
        sessions = list_all_sessions()
        if sessions:
            for s in sessions:
                status = "Complete" if s["completed"] else "In progress"
                st.write(f"**{s['evaluator']}** -- {s['progress']} -- {status}")

            st.divider()
            zip_data = build_sessions_zip()
            st.download_button(
                "Download ALL sessions (ZIP)",
                data=zip_data,
                file_name=f"eval_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
            )
        else:
            st.write("No sessions yet.")


def page_eval():
    """Main evaluation page."""
    session: EvaluationSession = st.session_state["session"]

    if session.is_complete:
        st.session_state["page"] = "done"
        st.rerun()
        return

    sample = session.get_current_sample()
    if sample is None:
        st.session_state["page"] = "done"
        st.rerun()
        return

    total = len(session.samples)
    done = len(session.evaluations)

    # --- Top bar ---
    st.progress(done / total)
    st.caption(f"Sample {done + 1} / {total}")

    # --- Layout ---
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Text")
        st.info(sample.text_content)

        st.subheader("Image")
        img_path = Path(sample.image_path)
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.warning(f"Image not found: {img_path.name}")

        st.subheader("Audio")
        audio_path = Path(sample.audio_path)
        if audio_path.exists():
            st.audio(str(audio_path))
        else:
            st.warning(f"Audio not found: {audio_path.name}")

    with right:
        st.subheader("Ratings")

        # --- Text-Image ---
        with st.expander("Text-Image rubric"):
            for score, desc in RUBRIC.text_image_rubric.items():
                st.markdown(f"**{score}** -- {desc}")
        ti = st.radio(
            "Text-Image coherence",
            options=[1, 2, 3, 4, 5],
            index=2,
            horizontal=True,
            key=f"ti_{session.current_index}",
        )

        # --- Text-Audio ---
        with st.expander("Text-Audio rubric"):
            for score, desc in RUBRIC.text_audio_rubric.items():
                st.markdown(f"**{score}** -- {desc}")
        ta = st.radio(
            "Text-Audio coherence",
            options=[1, 2, 3, 4, 5],
            index=2,
            horizontal=True,
            key=f"ta_{session.current_index}",
        )

        # --- Image-Audio ---
        with st.expander("Image-Audio rubric"):
            for score, desc in RUBRIC.image_audio_rubric.items():
                st.markdown(f"**{score}** -- {desc}")
        ia = st.radio(
            "Image-Audio coherence",
            options=[1, 2, 3, 4, 5],
            index=2,
            horizontal=True,
            key=f"ia_{session.current_index}",
        )

        # --- Overall ---
        with st.expander("Overall rubric"):
            for score, desc in RUBRIC.overall_rubric.items():
                st.markdown(f"**{score}** -- {desc}")
        overall = st.radio(
            "Overall coherence",
            options=[1, 2, 3, 4, 5],
            index=2,
            horizontal=True,
            key=f"ov_{session.current_index}",
        )

        st.divider()

        confidence = st.slider(
            "Confidence in your ratings",
            min_value=1, max_value=5, value=3,
            key=f"conf_{session.current_index}",
        )

        notes = st.text_area(
            "Notes (optional)",
            key=f"notes_{session.current_index}",
            height=80,
        )

        # Determine if this is a re-rating sample
        n_original = len(session.samples) - len(session.rerating_sample_ids)
        is_rerating = (
            sample.sample_id in session.rerating_sample_ids
            and session.current_index >= n_original
        )

        if st.button("Submit & Next", type="primary", use_container_width=True):
            evaluation = HumanEvaluation(
                sample_id=sample.sample_id,
                evaluator_id=session.evaluator_id,
                text_image_coherence=ti,
                text_audio_coherence=ta,
                image_audio_coherence=ia,
                overall_coherence=overall,
                confidence=confidence,
                notes=notes,
                is_rerating=is_rerating,
            )
            session.add_evaluation(evaluation)
            save_session(session)
            st.rerun()

    # Sidebar: session info + quit
    with st.sidebar:
        st.write(f"**Evaluator:** {session.evaluator_id}")
        st.write(f"**Session:** {session.session_id[:15]}...")
        st.write(f"**Progress:** {done}/{total} ({session.progress:.0f}%)")
        if st.button("Save & Exit"):
            save_session(session)
            del st.session_state["session"]
            st.session_state["page"] = "login"
            st.rerun()


def page_done():
    """Completion page with download options."""
    session: EvaluationSession = st.session_state["session"]

    st.title("Session Complete")
    st.success(f"All {len(session.evaluations)} evaluations submitted. Thank you!")

    # --- Storage reminder ---
    if HF_TOKEN:
        st.info("Your ratings have been automatically saved to cloud storage.")
    else:
        st.warning(
            "**Important:** Download your session file now. "
            "Data may be lost if the Space restarts."
        )

    n_original = len(session.samples) - len(session.rerating_sample_ids)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Primary ratings", n_original)
    with col2:
        st.metric("Re-ratings (reliability)", len(session.rerating_sample_ids))
    with col3:
        if session.started_at and session.completed_at:
            start = datetime.fromisoformat(session.started_at)
            end = datetime.fromisoformat(session.completed_at)
            elapsed = end - start
            minutes = elapsed.total_seconds() / 60
            st.metric("Time taken", f"{minutes:.1f} min")

    # Download session JSON
    session_path = SESSIONS_DIR / f"session_{session.session_id}.json"
    if session_path.exists():
        st.download_button(
            "Download session JSON",
            data=session_path.read_text(encoding="utf-8"),
            file_name=f"session_{session.session_id}.json",
            mime="application/json",
        )

    # Download ALL sessions as ZIP
    zip_data = build_sessions_zip()
    st.download_button(
        "Download ALL sessions (ZIP)",
        data=zip_data,
        file_name=f"eval_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
    )

    # Inter-rater overview
    all_sessions_info = list_all_sessions()
    completed = [s for s in all_sessions_info if s["completed"]]
    if len(completed) > 1:
        st.subheader("Multi-rater overview")
        st.write(f"{len(completed)} completed sessions found.")
        for s in completed:
            st.write(f"- **{s['evaluator']}** -- {s['progress']}")

    if st.button("Back to login"):
        del st.session_state["session"]
        st.session_state["page"] = "login"
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Coherence Evaluation",
        page_icon="\U0001F3AF",
        layout="wide",
    )

    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    page = st.session_state["page"]

    if page == "login" or "session" not in st.session_state:
        page_login()
    elif page == "eval":
        page_eval()
    elif page == "done":
        page_done()
    else:
        page_login()


if __name__ == "__main__":
    main()
