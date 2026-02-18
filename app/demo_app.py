"""
Multimodal Coherence AI — Interactive Demo

Live demonstration of the multimodal generation + coherence evaluation system.
Type a scene description and watch the system produce coherent text, image, and audio
with real-time MSCI scoring.

Supports two pipelines for both image and audio:
  - Retrieval: instant, matches from curated indexes
  - Generative: SDXL for images (~12s), AudioLDM 2 for audio (~8s)

Launch:
    streamlit run app/demo_app.py
    streamlit run app/demo_app.py --server.address 0.0.0.0  # LAN access
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

IMAGE_SIM_THRESHOLD = 0.20
AUDIO_SIM_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# Custom CSS — polished dark-mode design system
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

/* ── Global ── */
.block-container { padding-top: 1.2rem !important; max-width: 1200px; }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

/* ── Hero ── */
.hero-wrap {
    text-align: center;
    padding: 1.5rem 0 1rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.35rem;
}
.hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}
.hero-sub b { color: #c4b5fd; }

/* ── Prompt area ── */
.stTextArea textarea {
    border-radius: 14px !important;
    border: 1.5px solid rgba(129,140,248,0.25) !important;
    font-size: 0.95rem !important;
    padding: 0.9rem 1rem !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: rgba(129,140,248,0.6) !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.1) !important;
}

/* ── Chips / tags ── */
.chip-row { display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center; padding-top: 0.3rem; }
.chip {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.22rem 0.7rem; border-radius: 20px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em;
}
.chip-purple { background: rgba(129,140,248,0.14); color: #a5b4fc; }
.chip-pink   { background: rgba(244,114,182,0.14); color: #f9a8d4; }
.chip-green  { background: rgba(52,211,153,0.14);  color: #6ee7b7; }
.chip-amber  { background: rgba(251,191,36,0.12);  color: #fcd34d; }
.chip-dot { width: 6px; height: 6px; border-radius: 50%; }
.chip-dot-purple { background: #818cf8; }
.chip-dot-pink   { background: #f472b6; }
.chip-dot-green  { background: #34d399; }
.chip-dot-amber  { background: #fbbf24; }

/* ── Score cards ── */
.scores-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 0.5rem 0 0.3rem;
}
@media (max-width: 768px) { .scores-grid { grid-template-columns: repeat(2, 1fr); } }
.sc {
    border-radius: 16px; padding: 1.1rem 0.8rem; text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(10px);
    position: relative; overflow: hidden;
}
.sc::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 16px 16px 0 0;
}
.sc-high::before { background: linear-gradient(90deg, #10b981, #34d399); }
.sc-mid::before  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.sc-low::before  { background: linear-gradient(90deg, #ef4444, #fb7185); }
.sc-class::before { background: linear-gradient(90deg, #818cf8, #c084fc); }
.sc-lbl {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #64748b; margin-bottom: 0.4rem; font-weight: 600;
}
.sc-val {
    font-size: 1.9rem; font-weight: 700; line-height: 1.1;
    font-family: 'JetBrains Mono', monospace;
}
.sc-high .sc-val { color: #34d399; }
.sc-mid  .sc-val { color: #fbbf24; }
.sc-low  .sc-val { color: #fb7185; }
.sc-class .sc-val { font-size: 1.15rem; font-family: 'Inter', sans-serif; color: #c4b5fd; }
.sc-badge {
    display: inline-block; margin-top: 0.35rem; padding: 0.15rem 0.55rem;
    border-radius: 20px; font-size: 0.6rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
}
.sc-high .sc-badge { background: rgba(52,211,153,0.12); color: #34d399; }
.sc-mid  .sc-badge { background: rgba(251,191,36,0.12); color: #fbbf24; }
.sc-low  .sc-badge { background: rgba(251,113,133,0.12); color: #fb7185; }
.sc-class .sc-badge { background: rgba(196,181,253,0.12); color: #c4b5fd; }

/* ── Section labels ── */
.sec-label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em;
    font-weight: 700; margin-bottom: 0.6rem; padding-bottom: 0.35rem;
    border-bottom: 2px solid rgba(129,140,248,0.15); color: #818cf8;
}

/* ── Content card ── */
.text-card {
    border-radius: 14px; padding: 1.1rem 1.2rem;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    font-size: 0.9rem; line-height: 1.75; color: #cbd5e1;
}

/* ── Timing strip ── */
.timing {
    display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;
    padding: 0.4rem 0.8rem; border-radius: 10px;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04);
    font-size: 0.72rem; color: #64748b; margin: 0.4rem 0;
}
.timing span { white-space: nowrap; }
.timing .t-total { color: #a5b4fc; font-weight: 700; }
.timing .t-sep { color: rgba(255,255,255,0.08); }

/* ── Warning ── */
.warn-banner {
    border-radius: 12px; padding: 0.7rem 1rem; margin-bottom: 0.6rem;
    border-left: 3px solid #fbbf24; font-size: 0.82rem; color: #fcd34d;
    background: rgba(251,191,36,0.05);
}
.warn-banner b { color: #fde68a; }

/* ── Sim bars ── */
.sb { margin: 0.35rem 0; }
.sb-top { display: flex; justify-content: space-between; font-size: 0.68rem; color: #64748b; margin-bottom: 0.15rem; }
.sb-top .sb-v { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.sb-track { height: 5px; border-radius: 3px; background: rgba(255,255,255,0.05); overflow: hidden; }
.sb-fill { height: 100%; border-radius: 3px; }
.sbf-g { background: linear-gradient(90deg, #10b981, #34d399); }
.sbf-y { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.sbf-r { background: linear-gradient(90deg, #ef4444, #fb7185); }

/* ── Welcome ── */
.welcome {
    text-align: center; padding: 4rem 2rem; color: #475569;
}
.welcome-icons { font-size: 3.5rem; margin-bottom: 0.8rem; letter-spacing: 0.3rem; }
.welcome-text { font-size: 1.05rem; color: #64748b; }
.welcome-hint { font-size: 0.82rem; color: #475569; margin-top: 0.3rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
.sidebar-info {
    font-size: 0.72rem; color: #64748b; line-height: 1.6;
    padding: 0.8rem; border-radius: 10px;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04);
}
.sidebar-info b { color: #94a3b8; }
</style>
"""

# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------
EXAMPLE_PROMPTS = {
    "Nature": [
        "A peaceful forest at dawn with birdsong and morning mist",
        "A field of golden wheat under a warm summer sunset",
        "A dense jungle with exotic birds calling from the canopy",
    ],
    "Urban": [
        "A bustling city street at night with neon lights and traffic",
        "A quiet alley in an old town with distant footsteps echoing",
        "A cafe terrace on a busy boulevard with clinking glasses",
    ],
    "Water": [
        "Ocean waves crashing on a sandy beach at sunset",
        "Rain falling on a pond with ripples spreading across the surface",
        "A mountain stream flowing over rocks through a pine forest",
    ],
    "Mixed": [
        "A lighthouse on a cliff during a thunderstorm at night",
        "A bonfire on a beach with waves and guitar music at night",
        "A train passing through countryside with distant church bells",
    ],
}
DOMAIN_ICONS = {"nature": "🌿", "urban": "🏙️", "water": "🌊", "mixed": "🌐", "other": "📍"}


# ---------------------------------------------------------------------------
# Cached loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_coherence_engine():
    from src.coherence.coherence_engine import CoherenceEngine
    return CoherenceEngine(target_dim=512)

@st.cache_resource
def load_image_retriever():
    from src.generators.image.generator_improved import ImprovedImageRetrievalGenerator
    return ImprovedImageRetrievalGenerator(index_path="data/embeddings/image_index.npz", min_similarity=0.20)

@st.cache_resource
def load_audio_retriever():
    from src.generators.audio.retrieval import AudioRetrievalGenerator
    return AudioRetrievalGenerator(index_path="data/embeddings/audio_index.npz", min_similarity=0.10)


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def gen_text(prompt: str, mode: str) -> dict:
    if mode == "direct":
        from src.generators.text.generator import generate_text
        text = generate_text(prompt=prompt, use_ollama=True, deterministic=False)
        return {"text": text, "text_prompt": prompt, "image_prompt": prompt, "audio_prompt": prompt, "plan": None}
    plan = None
    if mode == "planner":
        from src.planner.unified_planner import UnifiedPlanner
        plan = UnifiedPlanner().plan(prompt)
    elif mode == "council":
        from src.planner.council import SemanticPlanningCouncil
        from src.planner.unified_planner import UnifiedPlannerLLM
        plan = SemanticPlanningCouncil(
            planner_a=UnifiedPlannerLLM(), planner_b=UnifiedPlannerLLM(), planner_c=UnifiedPlannerLLM(),
        ).run(prompt).merged_plan
    elif mode == "extended_prompt":
        from src.planner.extended_prompt_planner import ExtendedPromptPlanner
        plan = ExtendedPromptPlanner().plan(prompt)
    from src.planner.schema_to_text import plan_to_prompts
    prompts = plan_to_prompts(plan)
    from src.generators.text.generator import generate_text
    text = generate_text(prompt=prompts["text_prompt"], use_ollama=True, deterministic=False)
    return {
        "text": text, "text_prompt": prompts["text_prompt"],
        "image_prompt": prompts["image_prompt"], "audio_prompt": prompts["audio_prompt"],
        "shared_brief": prompts.get("shared_brief", ""), "plan": plan.model_dump() if hasattr(plan, "model_dump") else None,
    }


def retrieve_image(prompt: str) -> dict:
    r = load_image_retriever().retrieve(prompt)
    return {"path": r.image_path, "similarity": r.similarity, "domain": r.domain,
            "failed": r.retrieval_failed, "top_5": r.top_5, "backend": "retrieval"}


def gen_image_sdxl(prompt: str, seed: int = 42) -> dict:
    import torch
    from src.generators.image.generator_hybrid import HybridImageGenerator
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "runs" / "demo" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sdxl_{time.strftime('%Y%m%d_%H%M%S')}.png"
    g = HybridImageGenerator(force_sd=True, sd_model="sdxl", device=device, num_inference_steps=30, guidance_scale=7.5)
    try:
        r = g.generate(prompt=prompt, out_path=str(out_path), seed=seed)
        return {"path": r.image_path, "similarity": None, "domain": None,
                "failed": False, "top_5": None, "backend": r.backend, "seed": seed}
    finally:
        g.unload(); gc.collect()


def retrieve_audio(prompt: str) -> dict:
    r = load_audio_retriever().retrieve(prompt)
    return {"path": r.audio_path, "similarity": r.similarity, "failed": r.retrieval_failed, "top_5": r.top_5, "backend": "retrieval"}


def gen_audio_aldm(prompt: str, seed: int = 42) -> dict:
    import torch
    from src.generators.audio.generator import AudioGenerator
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "runs" / "demo" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"aldm2_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    g = AudioGenerator(device=device, force_audioldm=True)
    try:
        r = g.generate(prompt=prompt, out_path=str(out_path), seed=seed)
        return {"path": r.audio_path, "similarity": None, "failed": False,
                "top_5": None, "backend": r.backend, "seed": seed,
                "duration": r.duration_sec}
    finally:
        g.unload(); gc.collect()


def eval_coherence(text: str, image_path: str, audio_path: str) -> dict:
    return load_coherence_engine().evaluate(text=text, image_path=image_path, audio_path=audio_path)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _sc_cls(v: Optional[float]) -> str:
    if v is None: return ""
    if v >= 0.45: return "sc-high"
    if v >= 0.30: return "sc-mid"
    return "sc-low"

def _sc_badge(v: Optional[float]) -> str:
    if v is None: return ""
    if v >= 0.45: return "High"
    if v >= 0.30: return "Moderate"
    return "Low"

def score_card_html(label: str, value: Optional[float], is_class: bool = False) -> str:
    if is_class:
        badge_text = _sc_badge(value) or "N/A"
        val_display = f"{badge_text} Coherence"
        badge_html = f'<div class="sc-badge">MSCI {value:.3f}</div>' if value is not None else ""
        return (f'<div class="sc sc-class"><div class="sc-lbl">{label}</div>'
                f'<div class="sc-val">{val_display}</div>{badge_html}</div>')
    cls = _sc_cls(value)
    val_str = f"{value:.4f}" if value is not None else "—"
    badge = _sc_badge(value)
    badge_html = f'<div class="sc-badge">{badge}</div>' if badge else ""
    return (f'<div class="sc {cls}"><div class="sc-lbl">{label}</div>'
            f'<div class="sc-val">{val_str}</div>{badge_html}</div>')

def sim_bar_html(name: str, val: float, mx: float = 0.6) -> str:
    pct = min(val / mx * 100, 100)
    cls = "sbf-g" if val >= 0.35 else ("sbf-y" if val >= 0.20 else "sbf-r")
    return (f'<div class="sb"><div class="sb-top"><span>{name}</span>'
            f'<span class="sb-v">{val:.4f}</span></div>'
            f'<div class="sb-track"><div class="sb-fill {cls}" style="width:{pct}%"></div></div></div>')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Multimodal Coherence AI", page_icon="🎨", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Hero
    st.markdown(
        '<div class="hero-wrap">'
        '<div class="hero-title">Multimodal Coherence AI</div>'
        '<div class="hero-sub">Generate semantically coherent <b>text + image + audio</b> bundles '
        'and evaluate cross-modal alignment with the <b>MSCI</b> metric.</div>'
        '</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("#### Configuration")

        pipeline = st.radio("Pipeline", ["Retrieval", "Generative"], index=0, horizontal=True,
                            help="**Retrieval**: instant from curated index\n\n**Generative**: SDXL (image) + AudioLDM 2 (audio)")
        use_gen = pipeline == "Generative"

        mode = st.selectbox("Planning", ["direct", "planner", "council", "extended_prompt"],
                            format_func=lambda x: {"direct":"Direct","planner":"Planner","council":"Council (3-way)","extended_prompt":"Extended (3x)"}[x])

        st.divider()
        st.markdown("#### Examples")
        for dname, prompts in EXAMPLE_PROMPTS.items():
            icon = DOMAIN_ICONS.get(dname.lower(), "📍")
            with st.expander(f"{icon} {dname}"):
                for p in prompts:
                    if st.button(p, key=f"ex_{hash(p)}", use_container_width=True):
                        st.session_state["prompt_input"] = p

        st.divider()
        gen_models = "SDXL + AudioLDM 2" if use_gen else "CLIP + CLAP retrieval"
        st.markdown(
            f'<div class="sidebar-info">'
            f'<b>Text</b> Ollama / Qwen2:7b<br>'
            f'<b>Image</b> {"SDXL (Stable Diffusion XL)" if use_gen else "CLIP retrieval (57 images)"}<br>'
            f'<b>Audio</b> {"AudioLDM 2 (cvssp/audioldm2)" if use_gen else "CLAP retrieval (104 clips)"}<br><br>'
            f'<b>Metric</b> MSCI = 0.45 × s<sub>t,i</sub> + 0.45 × s<sub>t,a</sub>'
            f'</div>', unsafe_allow_html=True)

    # Prompt input
    default_prompt = st.session_state.get("prompt_input", "")
    prompt = st.text_area("Scene", value=default_prompt, height=80,
        placeholder="Describe a scene... e.g., 'A peaceful forest at dawn with birdsong and morning mist'",
        label_visibility="collapsed")

    # Button + chips
    bc1, bc2 = st.columns([1, 3])
    with bc1:
        go = st.button("Generate Bundle", type="primary", use_container_width=True, disabled=not prompt.strip())
    with bc2:
        pcls = "chip-pink" if use_gen else "chip-purple"
        pdot = "chip-dot-pink" if use_gen else "chip-dot-purple"
        plbl = "Generative" if use_gen else "Retrieval"
        mlbl = {"direct":"Direct","planner":"Planner","council":"Council","extended_prompt":"Extended"}[mode]
        st.markdown(
            f'<div class="chip-row">'
            f'<span class="chip {pcls}"><span class="chip-dot {pdot}"></span>{plbl}</span>'
            f'<span class="chip chip-green"><span class="chip-dot chip-dot-green"></span>{mlbl}</span>'
            f'</div>', unsafe_allow_html=True)

    # Welcome
    if not go and "last_result" not in st.session_state:
        st.markdown(
            '<div class="welcome">'
            '<div class="welcome-icons">🎨  🖼️  🔊</div>'
            '<div class="welcome-text">Enter a scene description and click <b>Generate Bundle</b></div>'
            '<div class="welcome-hint">or pick an example from the sidebar</div>'
            '</div>', unsafe_allow_html=True)
        return

    if go and prompt.strip():
        st.session_state["last_result"] = run_pipeline(prompt.strip(), mode, use_gen)

    if "last_result" in st.session_state:
        show_results(st.session_state["last_result"])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(prompt: str, mode: str, use_gen: bool) -> dict:
    R = {"use_gen": use_gen}
    t_all = time.time()

    # 1) Text
    with st.status("Generating text...", expanded=True) as s:
        t0 = time.time()
        try:
            R["text"] = gen_text(prompt, mode)
            R["t_text"] = time.time() - t0
            s.update(label=f"Text ready ({R['t_text']:.1f}s)", state="complete")
        except Exception as e:
            s.update(label=f"Text failed: {e}", state="error")
            R["text"] = {"text": f"[Failed: {e}]", "image_prompt": prompt, "audio_prompt": prompt}
            R["t_text"] = time.time() - t0

    ip = R["text"].get("image_prompt", prompt)
    ap = R["text"].get("audio_prompt", prompt)

    # 2) Image
    if use_gen:
        with st.status("Generating image with SDXL...", expanded=True) as s:
            t0 = time.time()
            try:
                R["image"] = gen_image_sdxl(ip)
                R["t_img"] = time.time() - t0
                s.update(label=f"Image generated — {R['image']['backend']} ({R['t_img']:.1f}s)", state="complete")
            except Exception as e:
                s.update(label=f"SDXL failed, falling back to retrieval", state="error")
                try: R["image"] = retrieve_image(ip); R["image"]["backend"] = "retrieval (fallback)"
                except: R["image"] = None
                R["t_img"] = time.time() - t0
    else:
        with st.status("Retrieving image...", expanded=True) as s:
            t0 = time.time()
            try:
                R["image"] = retrieve_image(ip)
                R["t_img"] = time.time() - t0
                f = R["image"].get("failed", False)
                lbl = f"Image retrieved (sim={R['image']['similarity']:.3f}, {R['t_img']:.1f}s)"
                if f: lbl += " — below threshold"
                s.update(label=lbl, state="complete" if not f else "error")
            except Exception as e:
                s.update(label=f"Image failed: {e}", state="error")
                R["image"] = None; R["t_img"] = time.time() - t0

    # 3) Audio
    if use_gen:
        with st.status("Generating audio with AudioLDM 2...", expanded=True) as s:
            t0 = time.time()
            try:
                R["audio"] = gen_audio_aldm(ap)
                R["t_aud"] = time.time() - t0
                s.update(label=f"Audio generated — {R['audio']['backend']} ({R['t_aud']:.1f}s)", state="complete")
            except Exception as e:
                s.update(label=f"AudioLDM 2 failed, falling back to retrieval", state="error")
                try: R["audio"] = retrieve_audio(ap)
                except: R["audio"] = None
                R["t_aud"] = time.time() - t0
    else:
        with st.status("Retrieving audio...", expanded=True) as s:
            t0 = time.time()
            try:
                R["audio"] = retrieve_audio(ap)
                R["t_aud"] = time.time() - t0
                f = R["audio"].get("failed", False)
                lbl = f"Audio retrieved (sim={R['audio']['similarity']:.3f}, {R['t_aud']:.1f}s)"
                if f: lbl += " — below threshold"
                s.update(label=lbl, state="complete" if not f else "error")
            except Exception as e:
                s.update(label=f"Audio failed: {e}", state="error")
                R["audio"] = None; R["t_aud"] = time.time() - t0

    # 4) Coherence
    with st.status("Evaluating coherence...", expanded=True) as s:
        t0 = time.time()
        try:
            imgp = R.get("image", {}).get("path") if R.get("image") else None
            audp = R.get("audio", {}).get("path") if R.get("audio") else None
            R["coherence"] = eval_coherence(R["text"]["text"], imgp, audp)
            R["t_eval"] = time.time() - t0
            msci = R["coherence"].get("scores", {}).get("msci")
            s.update(label=f"MSCI = {msci:.4f} ({R['t_eval']:.1f}s)", state="complete")
        except Exception as e:
            s.update(label=f"Eval failed: {e}", state="error")
            R["coherence"] = None; R["t_eval"] = time.time() - t0

    R["t_total"] = time.time() - t_all
    R["prompt"] = prompt; R["mode"] = mode
    return R


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def show_results(R: dict):
    coh = R.get("coherence")
    sc = coh.get("scores", {}) if coh else {}
    msci, st_i, st_a = sc.get("msci"), sc.get("st_i"), sc.get("st_a")

    # Score cards
    st.markdown('<div class="sec-label">Coherence Scores</div>', unsafe_allow_html=True)
    cards = (score_card_html("MSCI (Overall)", msci)
             + score_card_html("Text → Image", st_i)
             + score_card_html("Text → Audio", st_a)
             + score_card_html("Classification", msci, is_class=True))
    st.markdown(f'<div class="scores-grid">{cards}</div>', unsafe_allow_html=True)

    # Timing
    tt = R.get("t_total", 0)
    sep = '<span class="t-sep">|</span>'
    st.markdown(
        f'<div class="timing">'
        f'<span class="t-total">Total {tt:.1f}s</span>{sep}'
        f'<span>Text {R.get("t_text",0):.1f}s</span>{sep}'
        f'<span>Image {R.get("t_img",0):.1f}s</span>{sep}'
        f'<span>Audio {R.get("t_aud",0):.1f}s</span>{sep}'
        f'<span>Eval {R.get("t_eval",0):.1f}s</span>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Three columns
    ct, ci, ca = st.columns([1.15, 1, 0.85])

    with ct:
        st.markdown('<div class="sec-label">Generated Text</div>', unsafe_allow_html=True)
        txt = R.get("text", {}).get("text", "")
        st.markdown(f'<div class="text-card">{txt}</div>', unsafe_allow_html=True)

    with ci:
        st.markdown('<div class="sec-label">Image</div>', unsafe_allow_html=True)
        ii = R.get("image")
        if ii and ii.get("path"):
            ip = Path(ii["path"])
            bk = ii.get("backend", "retrieval")
            is_ret = "retrieval" in bk
            failed = ii.get("failed", False)
            sim = ii.get("similarity")

            if is_ret and failed:
                st.markdown(
                    f'<div class="warn-banner"><b>Below threshold</b> (sim={sim:.3f} &lt; {IMAGE_SIM_THRESHOLD}) '
                    f'— try <b>Generative</b> pipeline for this prompt.</div>', unsafe_allow_html=True)

            if ip.exists():
                st.image(str(ip), use_container_width=True)
                if is_ret:
                    dom = ii.get("domain", "other")
                    ic = DOMAIN_ICONS.get(dom, "📍")
                    st.caption(f"{ic} {dom} · sim **{sim:.3f}** · {ip.name}")
                else:
                    st.caption(f"**{bk}** · seed {ii.get('seed','—')} · {ip.name}")
        else:
            st.info("No image.")

    with ca:
        st.markdown('<div class="sec-label">Audio</div>', unsafe_allow_html=True)
        ai = R.get("audio")
        if ai and ai.get("path"):
            ap = Path(ai["path"])
            bk = ai.get("backend", "retrieval")
            is_ret = "retrieval" in bk
            sim = ai.get("similarity")
            failed = ai.get("failed", False)

            if is_ret and failed:
                st.markdown(
                    f'<div class="warn-banner"><b>Below threshold</b> (sim={sim:.3f} &lt; {AUDIO_SIM_THRESHOLD}).</div>',
                    unsafe_allow_html=True)

            if ap.exists():
                st.audio(str(ap))
                if is_ret:
                    st.caption(f"sim **{sim:.3f}** · {ap.name}")
                else:
                    st.caption(f"**{bk}** · seed {ai.get('seed','—')} · {ap.name}")
        else:
            st.info("No audio.")

    st.markdown("---")

    # Expandables
    with st.expander("Semantic Plan"):
        td = R.get("text", {})
        plan = td.get("plan")
        if plan:
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(f"**Scene** {plan.get('scene_summary','—')}")
                st.markdown(f"**Domain** {plan.get('domain','—')}")
                core = plan.get("core_semantics", {})
                st.markdown(f"**Setting** {core.get('setting','—')} · **Time** {core.get('time_of_day','—')} · **Weather** {core.get('weather','—')}")
                st.markdown(f"**Subjects** {', '.join(core.get('main_subjects',[]))}")
            with p2:
                st.markdown("**Image prompt**")
                st.code(td.get("image_prompt",""), language=None)
                st.markdown("**Audio prompt**")
                st.code(td.get("audio_prompt",""), language=None)
        else:
            st.write("Direct mode — no semantic plan.")

    with st.expander("Retrieval / Generation Details"):
        r1, r2 = st.columns(2)
        with r1:
            ii = R.get("image")
            bk = ii.get("backend","retrieval") if ii else "—"
            if ii and ii.get("top_5"):
                st.markdown("**Image — Top 5**")
                bars = "".join(sim_bar_html(n, s) for n, s in ii["top_5"])
                st.markdown(bars, unsafe_allow_html=True)
            elif ii and "retrieval" not in bk:
                st.markdown(f"**Image — Generated** `{bk}` (seed {ii.get('seed','—')})")
            else:
                st.write("No image data.")
        with r2:
            ai = R.get("audio")
            bk = ai.get("backend","retrieval") if ai else "—"
            if ai and ai.get("top_5"):
                st.markdown("**Audio — Top 5**")
                bars = "".join(sim_bar_html(n, s) for n, s in ai["top_5"])
                st.markdown(bars, unsafe_allow_html=True)
            elif ai and "retrieval" not in bk:
                st.markdown(f"**Audio — Generated** `{bk}` (seed {ai.get('seed','—')})")
            else:
                st.write("No audio data.")

    with st.expander("Full Coherence Report"):
        if coh:
            st.json(coh)
        else:
            st.write("No data.")


if __name__ == "__main__":
    main()
