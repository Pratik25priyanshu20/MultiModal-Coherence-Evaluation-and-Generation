"""
MultiModal Coherence AI — Results Dashboard

Professional visualization of experiment results (RQ1, RQ2, RQ3).
Box plots, effect sizes, significance testing, and dataset exploration.

Launch:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.statistical_analysis import (
    paired_ttest,
    descriptive_stats,
    bootstrap_ci,
    bootstrap_ci_diff,
    holm_bonferroni_correction,
    spearman_correlation,
)

# ---------------------------------------------------------------------------
# Theme & Colors
# ---------------------------------------------------------------------------
COLORS = {
    "baseline": "#2563EB",
    "wrong_image": "#DC2626",
    "wrong_audio": "#F59E0B",
    "direct": "#6B7280",
    "planner": "#2563EB",
    "council": "#7C3AED",
    "extended_prompt": "#059669",
    "primary": "#2563EB",
    "success": "#059669",
    "danger": "#DC2626",
    "warning": "#F59E0B",
    "muted": "#6B7280",
    "bg": "#FFFFFF",
    "bg_alt": "#F8FAFC",
}

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=13),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def styled_metric(label: str, value: str, delta: str = "", color: str = "primary"):
    """Render a styled metric card."""
    delta_html = ""
    if delta:
        delta_color = COLORS["success"] if delta.startswith("+") or delta.startswith("d") else COLORS["danger"]
        delta_html = f'<div style="font-size:0.85rem;color:{delta_color};margin-top:2px;">{delta}</div>'
    st.markdown(
        f"""<div style="background:{COLORS['bg_alt']};border-radius:12px;padding:16px 20px;
        border:1px solid #E2E8F0;text-align:center;">
        <div style="font-size:0.8rem;color:{COLORS['muted']};text-transform:uppercase;
        letter-spacing:0.05em;">{label}</div>
        <div style="font-size:1.8rem;font-weight:700;color:{COLORS[color]};margin:4px 0;">
        {value}</div>{delta_html}</div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_results(path: str):
    with open(path) as f:
        return json.load(f)


def find_results_files():
    """Scan runs/ for experiment result files."""
    runs_dir = PROJECT_ROOT / "runs"
    files = {}
    for pattern, label in [
        ("**/rq1_results.json", "RQ1"),
        ("**/rq2_results.json", "RQ2"),
    ]:
        for p in sorted(runs_dir.glob(pattern)):
            files[f"{label}: {p.parent.name}"] = str(p)
    return files


def aggregate_by_prompt(results, group_key, value_key="msci"):
    """Aggregate scores by prompt, averaging across seeds."""
    raw = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get(value_key) is not None and "error" not in r:
            raw[r[group_key]][r["prompt_id"]].append(r[value_key])

    aggregated = {}
    for group, prompt_scores in raw.items():
        aggregated[group] = {
            pid: float(np.mean(scores)) for pid, scores in prompt_scores.items()
        }

    if not aggregated:
        return {}

    all_groups = list(aggregated.keys())
    common = set(aggregated[all_groups[0]].keys())
    for g in all_groups[1:]:
        common &= set(aggregated[g].keys())
    common = sorted(common)

    return {g: [aggregated[g][pid] for pid in common] for g in all_groups}


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def page_overview():
    st.markdown("## Project Overview")
    st.markdown(
        "**MultiModal Coherence AI** evaluates cross-modal semantic alignment "
        "across text, image, and audio using the Multi-modal Semantic Coherence "
        "Index (MSCI)."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        styled_metric("Research Questions", "3", color="primary")
    with col2:
        styled_metric("Embedding Spaces", "2", delta="CLIP + CLAP", color="primary")
    with col3:
        # Count images
        img_dirs = [PROJECT_ROOT / "data/processed/images", PROJECT_ROOT / "data/wikimedia/images"]
        n_img = sum(
            len([f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
            for d in img_dirs if d.exists()
        )
        styled_metric("Images", str(n_img), color="primary")
    with col4:
        aud_dirs = [
            PROJECT_ROOT / "data/processed/audio",
            PROJECT_ROOT / "data/audiocaps/audio",
            PROJECT_ROOT / "data/freesound/audio",
        ]
        n_aud = sum(
            len([f for f in d.iterdir() if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}])
            for d in aud_dirs if d.exists()
        )
        styled_metric("Audio Files", str(n_aud), color="primary")

    st.divider()

    st.markdown("### MSCI Formula")
    st.latex(r"\text{MSCI} = w_{ti} \cdot s_{t,i} + w_{ta} \cdot s_{t,a} + w_{ia} \cdot s_{i,a}")
    st.markdown(
        "Where $s_{t,i}$ = text-image similarity (CLIP), "
        "$s_{t,a}$ = text-audio similarity (CLAP), "
        "$s_{i,a}$ = image-audio similarity (currently omitted — cross-space)."
    )

    st.markdown("### Research Questions")
    st.markdown("""
| RQ | Question | Status |
|----|----------|--------|
| RQ1 | Is MSCI sensitive to controlled semantic perturbations? | Tested |
| RQ2 | Does structured planning improve cross-modal alignment? | Tested |
| RQ3 | Does MSCI correlate with human coherence judgments? | Pending |
""")


# ---------------------------------------------------------------------------
# Page: RQ1
# ---------------------------------------------------------------------------

def page_rq1():
    st.markdown("## RQ1: MSCI Sensitivity to Perturbations")
    st.markdown(
        "Tests whether MSCI drops when images or audio are deliberately "
        "replaced with semantically mismatched content."
    )

    files = find_results_files()
    rq1_files = {k: v for k, v in files.items() if k.startswith("RQ1")}

    if not rq1_files:
        st.warning("No RQ1 results found. Run: `python scripts/run_rq1.py`")
        return

    selected = st.selectbox("Results file", list(rq1_files.keys()))
    data = load_results(rq1_files[selected])
    results = data.get("results", [])
    config = data.get("config", {})

    st.caption(
        f"{config.get('n_prompts', '?')} prompts | "
        f"Seeds: {config.get('seeds', '?')} | "
        f"{config.get('total_runs', '?')} total runs"
    )

    msci_by_cond = aggregate_by_prompt(results, "condition", "msci")
    if "baseline" not in msci_by_cond:
        st.error("No baseline results found.")
        return

    # --- Metric cards ---
    cols = st.columns(3)
    for i, cond in enumerate(["baseline", "wrong_image", "wrong_audio"]):
        if cond not in msci_by_cond:
            continue
        d = descriptive_stats(msci_by_cond[cond])
        delta = ""
        if cond != "baseline":
            diff = d["mean"] - descriptive_stats(msci_by_cond["baseline"])["mean"]
            delta = f"{diff:+.4f} vs baseline"
        with cols[i]:
            color = {"baseline": "primary", "wrong_image": "danger", "wrong_audio": "warning"}[cond]
            styled_metric(
                cond.replace("_", " ").title(),
                f"{d['mean']:.4f}",
                delta=delta,
                color=color,
            )

    st.markdown("")

    # --- Box plots ---
    fig = go.Figure()
    for cond in ["baseline", "wrong_image", "wrong_audio"]:
        if cond not in msci_by_cond:
            continue
        vals = msci_by_cond[cond]
        fig.add_trace(go.Box(
            y=vals,
            name=cond.replace("_", " ").title(),
            marker_color=COLORS[cond],
            boxmean="sd",
            jitter=0.3,
            pointpos=-1.5,
            boxpoints="all",
        ))

    fig.update_layout(
        title="MSCI Distribution by Condition",
        yaxis_title="MSCI Score",
        showlegend=False,
        **LAYOUT_DEFAULTS,
    )
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Statistical tests ---
    st.markdown("### Statistical Tests")

    test_data = []
    for pert in ["wrong_image", "wrong_audio"]:
        if pert not in msci_by_cond:
            continue
        result = paired_ttest(
            msci_by_cond["baseline"], msci_by_cond[pert],
            alpha=0.05, alternative="greater",
        )
        boot = bootstrap_ci_diff(
            msci_by_cond["baseline"], msci_by_cond[pert], paired=True,
        )
        test_data.append({
            "Comparison": f"Baseline vs {pert.replace('_', ' ').title()}",
            "t-stat": f"{result.statistic:.3f}",
            "p-value": f"{result.p_value:.6f}",
            "Cohen's d": f"{result.effect_size:.3f}",
            "Effect": result.interpretation,
            "95% CI (bootstrap)": f"[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]",
            "Sig.": "Yes" if result.significant else "No",
        })

    if test_data:
        st.dataframe(test_data, use_container_width=True, hide_index=True)

    # --- Forest plot (effect sizes) ---
    st.markdown("### Effect Size Forest Plot")
    fig_forest = go.Figure()

    for i, pert in enumerate(["wrong_image", "wrong_audio"]):
        if pert not in msci_by_cond:
            continue
        result = paired_ttest(
            msci_by_cond["baseline"], msci_by_cond[pert],
            alpha=0.05, alternative="greater",
        )
        boot = bootstrap_ci_diff(
            msci_by_cond["baseline"], msci_by_cond[pert], paired=True,
        )

        fig_forest.add_trace(go.Scatter(
            x=[boot["mean_diff"]],
            y=[pert.replace("_", " ").title()],
            error_x=dict(
                type="data",
                symmetric=False,
                array=[boot["ci_upper"] - boot["mean_diff"]],
                arrayminus=[boot["mean_diff"] - boot["ci_lower"]],
            ),
            mode="markers",
            marker=dict(size=12, color=COLORS[pert]),
            name=pert.replace("_", " ").title(),
            showlegend=False,
        ))

    fig_forest.add_vline(x=0, line_dash="dash", line_color=COLORS["muted"])
    fig_forest.update_layout(
        title="Mean MSCI Difference (Baseline - Perturbed)",
        xaxis_title="MSCI Difference (95% Bootstrap CI)",
        **LAYOUT_DEFAULTS,
    )
    fig_forest.update_xaxes(gridcolor="#E2E8F0")
    fig_forest.update_yaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig_forest, use_container_width=True)

    # --- Per-prompt paired differences ---
    st.markdown("### Per-Prompt Paired Differences")
    for pert in ["wrong_image", "wrong_audio"]:
        if pert not in msci_by_cond:
            continue
        diffs = np.array(msci_by_cond["baseline"]) - np.array(msci_by_cond[pert])

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=diffs,
            nbinsx=20,
            marker_color=COLORS[pert],
            opacity=0.8,
            name=pert.replace("_", " ").title(),
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color=COLORS["muted"])
        fig_hist.add_vline(x=float(np.mean(diffs)), line_color=COLORS["primary"],
                           annotation_text=f"Mean={np.mean(diffs):.4f}")
        fig_hist.update_layout(
            title=f"Baseline - {pert.replace('_', ' ').title()} (per prompt)",
            xaxis_title="MSCI Difference",
            yaxis_title="Count",
            showlegend=False,
            **LAYOUT_DEFAULTS,
        )
        fig_hist.update_xaxes(gridcolor="#E2E8F0")
        fig_hist.update_yaxes(gridcolor="#E2E8F0")
        st.plotly_chart(fig_hist, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: RQ2
# ---------------------------------------------------------------------------

def page_rq2():
    st.markdown("## RQ2: Planning Effect on Alignment")
    st.markdown(
        "Tests whether structured semantic planning improves MSCI "
        "compared to direct prompt-to-generation."
    )

    files = find_results_files()
    rq2_files = {k: v for k, v in files.items() if k.startswith("RQ2")}

    if not rq2_files:
        st.warning("No RQ2 results found. Run: `python scripts/run_rq2.py`")
        return

    selected = st.selectbox("Results file", list(rq2_files.keys()))
    data = load_results(rq2_files[selected])
    results = data.get("results", [])
    config = data.get("config", {})

    st.caption(
        f"{config.get('n_prompts', '?')} prompts | "
        f"Seeds: {config.get('seeds', '?')} | "
        f"Modes: {config.get('modes', '?')}"
    )

    msci_by_mode = aggregate_by_prompt(results, "mode", "msci")
    modes = sorted(msci_by_mode.keys())

    if not modes:
        st.error("No valid results found.")
        return

    # --- Metric cards ---
    cols = st.columns(len(modes))
    mode_colors = {"direct": "muted", "planner": "primary", "council": "primary", "extended_prompt": "success"}
    for i, mode in enumerate(modes):
        d = descriptive_stats(msci_by_mode[mode])
        delta = ""
        if mode != "direct" and "direct" in msci_by_mode:
            diff = d["mean"] - descriptive_stats(msci_by_mode["direct"])["mean"]
            delta = f"{diff:+.4f} vs direct"
        with cols[i]:
            styled_metric(
                mode.replace("_", " ").title(),
                f"{d['mean']:.4f}",
                delta=delta,
                color=mode_colors.get(mode, "primary"),
            )

    st.markdown("")

    # --- Box plots ---
    fig = go.Figure()
    for mode in modes:
        color = COLORS.get(mode, COLORS["primary"])
        fig.add_trace(go.Box(
            y=msci_by_mode[mode],
            name=mode.replace("_", " ").title(),
            marker_color=color,
            boxmean="sd",
            jitter=0.3,
            pointpos=-1.5,
            boxpoints="all",
        ))

    fig.update_layout(
        title="MSCI Distribution by Planning Mode",
        yaxis_title="MSCI Score",
        showlegend=False,
        **LAYOUT_DEFAULTS,
    )
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Statistical tests ---
    if "direct" in msci_by_mode:
        st.markdown("### Statistical Tests (vs Direct)")
        test_data = []
        for mode in modes:
            if mode == "direct":
                continue
            result = paired_ttest(
                msci_by_mode[mode], msci_by_mode["direct"],
                alpha=0.05, alternative="two-sided",
            )
            boot = bootstrap_ci_diff(
                msci_by_mode[mode], msci_by_mode["direct"], paired=True,
            )
            test_data.append({
                "Comparison": f"{mode.replace('_', ' ').title()} vs Direct",
                "t-stat": f"{result.statistic:.3f}",
                "p-value": f"{result.p_value:.6f}",
                "Cohen's d": f"{result.effect_size:.3f}",
                "Effect": result.interpretation,
                "95% CI (bootstrap)": f"[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]",
                "Sig.": "Yes" if result.significant else "No",
            })

        if test_data:
            st.dataframe(test_data, use_container_width=True, hide_index=True)

    # --- Sub-metric breakdown ---
    st.markdown("### Sub-metric Breakdown")
    sti_by_mode = aggregate_by_prompt(results, "mode", "st_i")
    sta_by_mode = aggregate_by_prompt(results, "mode", "st_a")

    fig_sub = make_subplots(rows=1, cols=2, subplot_titles=["Text-Image (st_i)", "Text-Audio (st_a)"])

    for mode in modes:
        color = COLORS.get(mode, COLORS["primary"])
        if mode in sti_by_mode:
            fig_sub.add_trace(go.Box(
                y=sti_by_mode[mode],
                name=mode.replace("_", " ").title(),
                marker_color=color,
                boxmean="sd",
                showlegend=True,
            ), row=1, col=1)
        if mode in sta_by_mode:
            fig_sub.add_trace(go.Box(
                y=sta_by_mode[mode],
                name=mode.replace("_", " ").title(),
                marker_color=color,
                boxmean="sd",
                showlegend=False,
            ), row=1, col=2)

    fig_sub.update_layout(
        title="Sub-metric Distributions",
        height=400,
        **LAYOUT_DEFAULTS,
    )
    fig_sub.update_yaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig_sub, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Dataset
# ---------------------------------------------------------------------------

def page_dataset():
    st.markdown("## Dataset Explorer")

    # Image index stats
    img_index = PROJECT_ROOT / "data/embeddings/image_index.npz"
    aud_index = PROJECT_ROOT / "data/embeddings/audio_index.npz"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Image Index")
        if img_index.exists():
            data = np.load(img_index, allow_pickle=True)
            ids = data["ids"].tolist()
            domains = data["domains"].tolist() if "domains" in data else []

            styled_metric("Total Images", str(len(ids)), color="primary")

            if domains:
                domain_counts = {}
                for d in domains:
                    domain_counts[d] = domain_counts.get(d, 0) + 1

                fig = go.Figure(data=[go.Pie(
                    labels=list(domain_counts.keys()),
                    values=list(domain_counts.values()),
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Set2,
                )])
                fig.update_layout(
                    title="Image Domain Distribution",
                    height=350,
                    **LAYOUT_DEFAULTS,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Show sample images
            with st.expander("Browse images"):
                img_dirs = [
                    PROJECT_ROOT / "data/processed/images",
                    PROJECT_ROOT / "data/wikimedia/images",
                ]
                all_imgs = []
                for d in img_dirs:
                    if d.exists():
                        all_imgs.extend(sorted([
                            f for f in d.iterdir()
                            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                        ]))
                if all_imgs:
                    page_size = 12
                    page_num = st.number_input("Page", 1, max((len(all_imgs) - 1) // page_size + 1, 1), 1, key="img_page")
                    start = (page_num - 1) * page_size
                    page_imgs = all_imgs[start:start + page_size]

                    img_cols = st.columns(4)
                    for idx, img in enumerate(page_imgs):
                        with img_cols[idx % 4]:
                            st.image(str(img), caption=img.stem[:30], use_container_width=True)
        else:
            st.warning("No image index found. Run: `python scripts/build_embedding_indexes.py`")

    with col2:
        st.markdown("### Audio Index")
        if aud_index.exists():
            data = np.load(aud_index, allow_pickle=True)
            ids = data["ids"].tolist()
            domains = data["domains"].tolist() if "domains" in data else []

            styled_metric("Total Audio Files", str(len(ids)), color="primary")

            if domains:
                domain_counts = {}
                for d in domains:
                    domain_counts[d] = domain_counts.get(d, 0) + 1

                fig = go.Figure(data=[go.Pie(
                    labels=list(domain_counts.keys()),
                    values=list(domain_counts.values()),
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Set2,
                )])
                fig.update_layout(
                    title="Audio Domain Distribution",
                    height=350,
                    **LAYOUT_DEFAULTS,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Show sample audio
            with st.expander("Browse audio"):
                aud_dirs = [
                    PROJECT_ROOT / "data/processed/audio",
                    PROJECT_ROOT / "data/audiocaps/audio",
                    PROJECT_ROOT / "data/freesound/audio",
                ]
                all_auds = []
                for d in aud_dirs:
                    if d.exists():
                        all_auds.extend(sorted([
                            f for f in d.iterdir()
                            if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
                        ]))
                if all_auds:
                    page_size = 10
                    page_num = st.number_input("Page", 1, max((len(all_auds) - 1) // page_size + 1, 1), 1, key="aud_page")
                    start = (page_num - 1) * page_size
                    page_auds = all_auds[start:start + page_size]

                    for aud in page_auds:
                        st.markdown(f"**{aud.stem[:40]}**")
                        st.audio(str(aud))
        else:
            st.warning("No audio index found. Run: `python scripts/build_embedding_indexes.py`")

    # Embedding space visualization
    st.divider()
    st.markdown("### Embedding Space Similarity Distribution")

    if img_index.exists():
        img_data = np.load(img_index, allow_pickle=True)
        embs = img_data["embs"].astype("float32")
        if len(embs) > 1:
            # Compute pairwise cosine similarities (sample if large)
            n = min(len(embs), 100)
            sample_embs = embs[:n]
            norms = np.linalg.norm(sample_embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            normalized = sample_embs / norms
            sim_matrix = normalized @ normalized.T
            # Extract upper triangle
            triu_idx = np.triu_indices(n, k=1)
            sims = sim_matrix[triu_idx]

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=sims, nbinsx=50, marker_color=COLORS["primary"], opacity=0.7,
                name="Image-Image",
            ))
            fig.update_layout(
                title="Pairwise Image Embedding Similarities (CLIP)",
                xaxis_title="Cosine Similarity",
                yaxis_title="Count",
                **LAYOUT_DEFAULTS,
            )
            fig.update_xaxes(gridcolor="#E2E8F0")
            fig.update_yaxes(gridcolor="#E2E8F0")
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Export
# ---------------------------------------------------------------------------

def page_export():
    st.markdown("## Export Results")
    st.markdown("Generate publication-ready tables and figures.")

    files = find_results_files()
    if not files:
        st.warning("No results found.")
        return

    selected = st.selectbox("Results file", list(files.keys()))
    data = load_results(files[selected])
    results = data.get("results", [])

    is_rq1 = "RQ1" in selected or "condition" in (results[0] if results else {})
    group_key = "condition" if is_rq1 else "mode"

    msci_data = aggregate_by_prompt(results, group_key, "msci")
    groups = sorted(msci_data.keys())

    # LaTeX table
    st.markdown("### LaTeX Table")
    latex_rows = []
    for g in groups:
        d = descriptive_stats(msci_data[g])
        ci = f"[{d['ci_lower_95']:.4f}, {d['ci_upper_95']:.4f}]"
        latex_rows.append(
            f"    {g.replace('_', ' ').title()} & {d['n']} & "
            f"{d['mean']:.4f} & {d['std']:.4f} & {ci} \\\\"
        )

    latex_table = (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{MSCI descriptive statistics with 95\\% bootstrap CIs}\n"
        "\\begin{tabular}{lcccc}\n\\hline\n"
        f"    {'Condition' if is_rq1 else 'Mode'} & N & Mean & SD & 95\\% CI \\\\\n\\hline\n"
        + "\n".join(latex_rows) +
        "\n\\hline\n\\end{tabular}\n\\end{table}"
    )
    st.code(latex_table, language="latex")

    # CSV download
    st.markdown("### CSV Download")
    csv_rows = ["prompt_id,seed," + group_key + ",msci,st_i,st_a"]
    for r in results:
        if "error" not in r and r.get("msci") is not None:
            csv_rows.append(
                f"{r.get('prompt_id','')},{r.get('seed','')},{r.get(group_key,'')},"
                f"{r['msci']:.6f},{r.get('st_i', '')},{r.get('st_a', '')}"
            )
    csv_text = "\n".join(csv_rows)
    st.download_button("Download CSV", csv_text, file_name="results.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="MultiModal Coherence AI",
        page_icon="M",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {background: #F8FAFC;}
        .stTabs [data-baseweb="tab-list"] {gap: 8px;}
        .stTabs [data-baseweb="tab"] {
            padding: 8px 20px; border-radius: 8px;
            font-weight: 500;
        }
        .stDataFrame {border-radius: 8px; overflow: hidden;}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("# MultiModal Coherence AI")
        st.caption("Results Dashboard")
        st.divider()
        page = st.radio(
            "Navigation",
            ["Overview", "RQ1: Sensitivity", "RQ2: Planning", "Dataset", "Export"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Launch human eval:")
        st.code("streamlit run app/human_eval_app.py", language="bash")

    # Page routing
    if page == "Overview":
        page_overview()
    elif page == "RQ1: Sensitivity":
        page_rq1()
    elif page == "RQ2: Planning":
        page_rq2()
    elif page == "Dataset":
        page_dataset()
    elif page == "Export":
        page_export()


if __name__ == "__main__":
    main()
