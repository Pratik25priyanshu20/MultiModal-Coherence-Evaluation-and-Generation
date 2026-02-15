"""
Streamlit Human Evaluation App for RQ3

Browser-based interface for collecting human coherence judgments on
multimodal bundles (text + image + audio). Supports multiple evaluators,
session persistence, blind evaluation with re-rating, and admin dashboard.

Launch:
    streamlit run app/human_eval_app.py
    streamlit run app/human_eval_app.py --server.address 0.0.0.0  # LAN access
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PATH = PROJECT_ROOT / "runs" / "rq3" / "rq3_samples.json"
SESSIONS_DIR = PROJECT_ROOT / "runs" / "rq3" / "sessions"

# Add project root to sys.path so we can import src.*
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.human_eval_schema import (
    CoherenceRubric,
    EvaluationSample,
    EvaluationSession,
    HumanEvaluation,
)

RUBRIC = CoherenceRubric()
RERATING_FRACTION = 0.20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_rq3_samples() -> List[EvaluationSample]:
    """Load the 30 pre-selected RQ3 samples from JSON."""
    with SAMPLES_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for s in data["samples"]:
        samples.append(EvaluationSample(
            sample_id=s["sample_id"],
            text_content=s["prompt_text"],
            image_path=s["image_path"],
            audio_path=s["audio_path"],
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
    """Build a new EvaluationSession for this evaluator."""
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
    """Persist session to disk."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"session_{session.session_id}.json"
    session.save(path)


def find_existing_sessions(evaluator_name: str) -> List[Path]:
    """Find all session files for an evaluator."""
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
    """List summary info for all sessions on disk."""
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


# ---------------------------------------------------------------------------
# Streamlit pages
# ---------------------------------------------------------------------------

def page_login():
    """Login / session management page."""
    st.title("Multimodal Coherence Evaluation")
    st.markdown("Rate the coherence of text + image + audio bundles.")

    evaluator_name = st.text_input("Your name", key="login_name")

    if not evaluator_name:
        st.info("Enter your name to begin.")
        # Show admin overview
        with st.expander("All sessions (admin)"):
            sessions = list_all_sessions()
            if sessions:
                for s in sessions:
                    status = "Complete" if s["completed"] else "In progress"
                    st.write(f"**{s['evaluator']}** — {s['progress']} — {status}")
            else:
                st.write("No sessions yet.")
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
                # Resume the most recent session
                session = EvaluationSession.load(existing[-1])
                st.session_state["session"] = session
                st.session_state["page"] = "eval"
                st.rerun()
        else:
            st.button("Resume Session", disabled=True, use_container_width=True,
                       help="No existing session found for this name")

    if existing:
        st.caption(f"Found {len(existing)} existing session(s) for **{evaluator_name}**.")

    # Admin overview
    with st.expander("All sessions (admin)"):
        sessions = list_all_sessions()
        if sessions:
            for s in sessions:
                status = "Complete" if s["completed"] else "In progress"
                st.write(f"**{s['evaluator']}** — {s['progress']} — {status}")
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
                st.markdown(f"**{score}** — {desc}")
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
                st.markdown(f"**{score}** — {desc}")
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
                st.markdown(f"**{score}** — {desc}")
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
                st.markdown(f"**{score}** — {desc}")
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
    """Completion / admin dashboard page."""
    session: EvaluationSession = st.session_state["session"]

    st.title("Session Complete")
    st.success(f"All {len(session.evaluations)} evaluations submitted. Thank you!")

    n_original = len(session.samples) - len(session.rerating_sample_ids)
    st.metric("Primary ratings", n_original)
    st.metric("Re-ratings (reliability)", len(session.rerating_sample_ids))

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

    # Inter-rater overview (if multiple sessions exist)
    all_sessions_info = list_all_sessions()
    completed = [s for s in all_sessions_info if s["completed"]]
    if len(completed) > 1:
        st.subheader("Multi-rater overview")
        st.write(f"{len(completed)} completed sessions found.")
        for s in completed:
            st.write(f"- **{s['evaluator']}** — {s['progress']}")

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
        page_icon="🎯",
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
