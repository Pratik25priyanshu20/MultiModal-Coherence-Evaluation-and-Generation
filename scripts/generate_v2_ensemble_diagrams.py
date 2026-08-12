#!/usr/bin/env python3
"""
Generate publication-quality architecture diagrams for:
  1. cMSCI v2 (Gemini unified space)
  2. Ensemble (v1 + v2 fusion)

Matches the visual style of fig_architecture.pdf (v1 diagram).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# ── Shared Colors ─────────────────────────────────────────────
C_GEMINI  = "#1A73E8"   # Gemini blue
C_STAGE   = "#F5F5F5"
C_BORDER  = "#C8C8C8"
C_GRAM    = "#2D8B57"
C_CALIB   = "#555555"
C_NEG     = "#6B5B95"
C_MRL     = "#E8710A"   # Matryoshka orange
C_OUT     = "#B8352D"
C_INPUT   = "#E0E0E0"
C_TEXT    = "#1A1A1A"
C_SUB     = "#777777"
C_ARR     = "#444444"
# Ensemble-specific
C_V1      = "#3B7DD8"   # v1 blue (CLIP heritage)
C_V2      = "#1A73E8"   # v2 blue (Gemini)
C_ENS     = "#D4382D"   # Ensemble red
C_CLIP    = "#3B7DD8"
C_CLAP    = "#D4722C"
C_EXMCR   = "#1E8C5E"
C_PROB    = "#7B4FA0"


# ── Shared Helpers ────────────────────────────────────────────

def _stage_bg(ax, x, y, w, h):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08",
        facecolor=C_STAGE, edgecolor=C_BORDER, linewidth=0.6, zorder=1)
    ax.add_patch(p)

def _stage_label(ax, x, y, num, text):
    ax.text(x, y, f"Stage {num}:  {text}",
            fontsize=7, fontweight="bold", color=C_SUB, va="center", zorder=5)

def _box(ax, x, y, w, h, label, color, text_color="white",
         fontsize=8.5, sublabel=None, sub_fs=7, lw=1.0):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04",
        facecolor=color, edgecolor=color, linewidth=lw, zorder=3)
    ax.add_patch(p)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.12, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
        ax.text(x + w/2, y + h/2 - 0.12, sublabel, ha="center", va="center",
                fontsize=sub_fs, color=text_color, zorder=4, alpha=0.85)
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)

def _arr(ax, x1, y1, x2, y2, color=C_ARR, lw=1.0):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", color=color,
        linewidth=lw, zorder=2, mutation_scale=10)
    ax.add_patch(a)

def _tag(ax, x, y, text, color, fs=5.5):
    ax.text(x, y, text, fontsize=fs, color=color, ha="center", va="center",
            zorder=5, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.06", facecolor="white",
                      edgecolor=color, alpha=0.85, linewidth=0.4))


# ══════════════════════════════════════════════════════════════
#  DIAGRAM 1:  cMSCI v2  (Gemini Unified Space)
# ══════════════════════════════════════════════════════════════

def draw_v2_architecture():
    W, H = 10.0, 16.0
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    cx = W / 2
    margin = 0.4
    full_w = W - 2 * margin

    # ── TITLE ─────────────────────────────────────────────────
    ax.text(cx, 15.65, "cMSCI v2 Pipeline Architecture",
            ha="center", fontsize=14, fontweight="bold", color=C_TEXT,
            fontfamily="serif")
    ax.text(cx, 15.35,
            "Gemini Embedding 2 — Unified 3072-d Space",
            ha="center", fontsize=8, color=C_SUB, fontstyle="italic")

    # ── INPUTS ────────────────────────────────────────────────
    iw, ih = 1.9, 0.38
    iy = 14.85
    i_centers = [1.5, 5.0, 8.5]
    for lbl, xc in zip(["Text", "Image", "Audio"], i_centers):
        _box(ax, xc - iw/2, iy, iw, ih, lbl, C_INPUT, C_TEXT, fontsize=9.5, lw=0.5)

    # ── STAGE 0: GEMINI ENCODER (unified) ─────────────────────
    s0y = 13.40
    _stage_label(ax, margin + 0.1, 14.48, 0, "Unified Embedding Extraction")
    _stage_bg(ax, margin, s0y, full_w, 1.0)

    # One big encoder box spanning the full width
    ew, eh = 8.4, 0.50
    ey = s0y + 0.10
    _box(ax, cx - ew/2, ey, ew, eh, "Gemini Embedding 2", C_GEMINI,
         fontsize=9, sublabel="gemini-embedding-2-preview  ·  3072-d  ·  unified text / image / audio",
         sub_fs=6)

    # "One space" annotation
    ax.text(cx, s0y + 0.85,
            "Single unified space — all pairwise + 3-way comparisons are direct",
            fontsize=6, color=C_GEMINI, ha="center", fontstyle="italic", zorder=5)

    e_cx = cx

    # Input → Encoder
    for ic in i_centers:
        _arr(ax, ic, iy, e_cx, ey + eh, C_GEMINI, 0.8)

    # 3 output embeddings fan out
    out_pts = [2.0, 5.0, 8.0]
    out_labels = [r"$\mathbf{e}_{text}$", r"$\mathbf{e}_{img}$", r"$\mathbf{e}_{aud}$"]
    for xc, lbl in zip(out_pts, out_labels):
        ax.text(xc, s0y - 0.12, lbl, fontsize=7, color=C_GEMINI,
                ha="center", va="center", zorder=5, fontweight="bold")
        _arr(ax, e_cx + (xc - e_cx) * 0.2, ey, xc, s0y + 0.02, C_GEMINI, 0.6)

    # ── STAGE 1: GRAMIAN VOLUME ───────────────────────────────
    s1y = 11.65
    _stage_label(ax, margin + 0.1, 12.95, 1, "Gramian Volume Geometry")
    _stage_bg(ax, margin, s1y, full_w, 1.20)

    gw_2d = 2.6
    gh = 0.42
    gy = s1y + 0.10

    # Three 2D Gramian boxes
    g2d_x = [0.55, 3.65, 6.75]
    g2d_labels = ["Gram TI", "Gram TA", "Gram IA"]
    g2d_subs = [
        r"$V_{ti} = |\sin\theta_{ti}|$",
        r"$V_{ta} = |\sin\theta_{ta}|$",
        r"$V_{ia} = |\sin\theta_{ia}|$",
    ]
    g2d_cx = []
    for x, lbl, sub in zip(g2d_x, g2d_labels, g2d_subs):
        _box(ax, x, gy, gw_2d, gh, lbl, C_GRAM, fontsize=7.5, sublabel=sub, sub_fs=6)
        g2d_cx.append(x + gw_2d / 2)

    # Exact 3D Gramian
    gy3 = gy + 0.55
    _box(ax, cx - 3.5, gy3, 7.0, 0.42, "Exact 3D Gramian Volume", C_GRAM,
         fontsize=8,
         sublabel=r"$V_{tia} = \sqrt{1 - \cos^2 a - \cos^2 b - \cos^2 c + 2\cos a\cos b\cos c}$",
         sub_fs=6)

    ax.text(cx + 3.85, gy3 + 0.21, "no approximation!", fontsize=5.5,
            color="#FFD700", ha="center", va="center", zorder=5,
            fontweight="bold", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.04", facecolor=C_GRAM,
                      edgecolor="#FFD700", linewidth=0.5, alpha=0.9))

    # Coherence annotation
    ax.text(cx, s1y + 1.12,
            r"Coherence:  $c = 1 - V$    (0 = orthogonal,  1 = aligned)",
            fontsize=6, color=C_GRAM, ha="center", fontstyle="italic", zorder=5)

    # Stage 0 → Stage 1
    for gc in g2d_cx:
        _arr(ax, gc, s0y - 0.05, gc, gy3 + gh, C_ARR, 0.6)

    # ── STAGE 2: Z-CALIBRATION ────────────────────────────────
    s2y = 10.30
    _stage_label(ax, margin + 0.1, 11.35, 2, "Z-Score Calibration (Gemini-specific)")
    _stage_bg(ax, margin, s2y, full_w, 0.95)

    zw = 2.6
    zh = 0.42
    zy = s2y + 0.10
    z_x = [0.55, 3.65, 6.75]
    z_labels = [r"$z_{ti}$", r"$z_{ta}$", r"$z_{ia}$"]
    z_subs = [
        r"$(c_{ti} - \mu_{ti}^{G}) / \sigma_{ti}^{G}$",
        r"$(c_{ta} - \mu_{ta}^{G}) / \sigma_{ta}^{G}$",
        r"$(c_{ia} - \mu_{ia}^{G}) / \sigma_{ia}^{G}$",
    ]
    z_cx = []
    for x, lbl, sub in zip(z_x, z_labels, z_subs):
        _box(ax, x, zy, zw, zh, lbl, C_CALIB, fontsize=8.5, sublabel=sub, sub_fs=6)
        z_cx.append(x + zw / 2)

    # Strike through z_ia to show w_ia=0
    ax.plot([z_x[2] + 0.1, z_x[2] + zw - 0.1],
            [zy + zh/2, zy + zh/2], color="#FF4444", lw=2.0, zorder=6, alpha=0.6)
    ax.text(z_x[2] + zw/2, zy - 0.12, r"$w_{ia} = 0$  (std=0.011, noise)",
            fontsize=5, color="#FF4444", ha="center", fontstyle="italic", zorder=5)

    # Stage 1 → Stage 2
    for gc, zc in zip(g2d_cx, z_cx):
        _arr(ax, gc, gy, zc, zy + zh, C_ARR, 0.6)

    # ── STAGES 3 + 4: CONTRASTIVE + MATRYOSHKA ───────────────
    br_top = 9.90
    br_bot = 6.85
    br_h = br_top - br_bot
    col_w = 4.3
    col_gap = 0.6
    col_x = [margin, margin + col_w + col_gap]

    # ── STAGE 3: CONTRASTIVE MARGIN (left) ────────────────────
    _stage_label(ax, col_x[0] + 0.1, br_top + 0.12, 3, "3-Channel Contrastive Margins")
    _stage_bg(ax, col_x[0], br_bot, col_w, br_h)

    s3_cx = col_x[0] + col_w / 2

    _tag(ax, s3_cx, 9.60, "from Stage 1:  V_ti, V_ta, V_ia (all in same space)", C_GRAM, 5)

    _box(ax, col_x[0] + 0.10, 9.00, col_w - 0.20, 0.42, "Hard Negative Mining", C_NEG,
         fontsize=7, sublabel="k = 5 per channel, Gemini indexes", sub_fs=5.5)

    # Per-channel margins
    _box(ax, col_x[0] + 0.10, 8.05, col_w - 0.20, 0.80, "", C_NEG, fontsize=7)
    ax.text(s3_cx, 8.70, "Per-Channel Margins (unified space)", fontsize=6.5,
            fontweight="bold", color="white", ha="center", va="center", zorder=4)
    ax.text(s3_cx, 8.48,
            r"$m_{ti} = \bar{V}_{ti}^{neg} - V_{ti}^{*}$"
            r"     "
            r"$m_{ta} = \bar{V}_{ta}^{neg} - V_{ta}^{*}$",
            fontsize=5.5, color="white", ha="center", va="center", zorder=4)
    ax.text(s3_cx, 8.28,
            r"$m_{ia} = \bar{V}_{ia}^{neg} - V_{ia}^{*}$"
            r"   $\rightarrow$   "
            r"$m = w_{ti} m_{ti} + (1-w_{ti}) m_{ta}$",
            fontsize=5.5, color="white", ha="center", va="center", zorder=4)
    ax.text(s3_cx, 8.10, r"$\alpha = 0$  (margins not needed in unified space)",
            fontsize=5, color="#FFD700", ha="center", fontstyle="italic", zorder=4)

    _arr(ax, s3_cx, 9.00, s3_cx, 8.85, C_NEG, 0.5)

    # Output
    _box(ax, col_x[0] + 0.10, 7.20, col_w - 0.20, 0.52, "", C_NEG, fontsize=7)
    ax.text(s3_cx, 7.55, "Result", fontsize=6.5,
            fontweight="bold", color="white", ha="center", va="center", zorder=4)
    ax.text(s3_cx, 7.34,
            r"$\alpha \cdot m = 0$  (contrastive vanishes)",
            fontsize=5.5, color="#FFD700", ha="center", va="center", zorder=4,
            fontweight="bold")

    _arr(ax, s3_cx, 8.05, s3_cx, 7.72, C_NEG, 0.5)

    # ── STAGE 4: MATRYOSHKA ADAPTIVE WEIGHTING (right) ────────
    _stage_label(ax, col_x[1] + 0.1, br_top + 0.12, 4, "Matryoshka Adaptive Weighting")
    _stage_bg(ax, col_x[1], br_bot, col_w, br_h)

    s4_cx = col_x[1] + col_w / 2

    _tag(ax, s4_cx, 9.60, "from Stage 0:  full 3072-d embeddings", C_GEMINI, 5)

    # MRL truncation
    _box(ax, col_x[1] + 0.10, 9.00, col_w - 0.20, 0.42,
         "Matryoshka Truncation", C_MRL,
         fontsize=7, sublabel="768-d  /  1536-d  /  3072-d  (re-normalize)", sub_fs=5.5)

    # Scale consistency
    _box(ax, col_x[1] + 0.10, 8.22, col_w - 0.20, 0.62, "", C_MRL, fontsize=7)
    ax.text(s4_cx, 8.68, "Scale Consistency (training-free!)", fontsize=6.5,
            fontweight="bold", color="white", ha="center", va="center", zorder=4)
    ax.text(s4_cx, 8.48,
            r"$c_d = \mathrm{gram\_coherence}(\mathbf{e}[:d])$   for  $d \in \{768, 1536, 3072\}$",
            fontsize=5.5, color="white", ha="center", va="center", zorder=4)
    ax.text(s4_cx, 8.30,
            r"$\mathrm{consistency} = 1 - \mathrm{std}(c_d) / \mathrm{mean}(c_d)$",
            fontsize=5.5, color="white", ha="center", va="center", zorder=4)

    _arr(ax, s4_cx, 9.00, s4_cx, 8.84, C_MRL, 0.5)

    # Adaptive weights
    _box(ax, col_x[1] + 0.10, 7.20, col_w - 0.20, 0.82, "", C_MRL, fontsize=7)
    ax.text(s4_cx, 7.88, "Adaptive Channel Weights", fontsize=6.5,
            fontweight="bold", color="white", ha="center", va="center", zorder=4)
    ax.text(s4_cx, 7.68,
            r"$w_{ti}^{mrl} = c_{ti} / (c_{ti} + c_{ta})$",
            fontsize=5.5, color="white", ha="center", va="center", zorder=4)
    ax.text(s4_cx, 7.50,
            r"$w_{ti}^{final} = (1-\gamma_{mrl}) w_{ti}^{base}"
            r" + \gamma_{mrl} w_{ti}^{mrl}$",
            fontsize=5.5, color="white", ha="center", va="center", zorder=4)
    ax.text(s4_cx, 7.32,
            r"$z_{2d}^{adapt} = w_{ti}^{f} z_{ti} + (1-w_{ti}^{f}) z_{ta}$",
            fontsize=5.5, color="#FFE0B2", ha="center", va="center",
            zorder=4, fontstyle="italic")

    _arr(ax, s4_cx, 8.22, s4_cx, 8.02, C_MRL, 0.5)

    # Stage 2 → branches
    _arr(ax, z_cx[0], zy, s3_cx, br_top, C_GRAM, 0.7)
    _arr(ax, z_cx[1], zy, s4_cx, br_top, C_CALIB, 0.7)

    # ── OUTPUT: cMSCI v2 ──────────────────────────────────────
    out_y = 4.80
    out_h = 1.70
    _stage_bg(ax, margin, out_y, full_w, out_h)
    _box(ax, margin + 0.1, out_y + 0.10, full_w - 0.2, out_h - 0.20, "", C_OUT,
         fontsize=12, lw=1.5)

    ax.text(cx, out_y + out_h - 0.30, "cMSCI v2 Score", fontsize=12, fontweight="bold",
            color="white", ha="center", va="center", zorder=4, fontfamily="serif")

    ax.text(cx, out_y + out_h - 0.65,
            r"$\mathrm{cMSCI}_{v2} = \sigma\left("
            r"z_{2d}^{adapt}"
            r" + \alpha \cdot m"
            r" + w_{compl} \cdot z_{compl}"
            r"\right)$",
            fontsize=9, color="white", ha="center", va="center", zorder=4)

    ax.text(cx, out_y + out_h - 1.0,
            r"Simplifies to:   $\mathrm{cMSCI}_{v2} = \sigma\left("
            r"w_{ti}^{f} z_{ti} + (1-w_{ti}^{f}) z_{ta}\right)$",
            fontsize=7.5, color="#FFD700", ha="center", va="center", zorder=4,
            fontweight="bold")

    ax.text(cx, out_y + out_h - 1.30,
            r"$w_{ti}^{base}=0.90$   $\gamma_{mrl}=0.6$   $\alpha=0$"
            r"   $w_{compl}=0$   $w_{ia}=0$",
            fontsize=6.5, color="#F5C6C6", ha="center", va="center", zorder=4)

    # Branches → Output
    _arr(ax, s3_cx, br_bot, cx - 1.5, out_y + out_h, C_NEG, 1.0)
    _arr(ax, s4_cx, br_bot, cx + 1.5, out_y + out_h, C_MRL, 1.0)

    # Arrow labels
    ax.text(s3_cx - 0.2, 6.68, r"$\alpha \cdot m = 0$", fontsize=6.5, color=C_NEG,
            ha="center", va="center", zorder=5)
    ax.text(s4_cx + 0.2, 6.68, r"$z_{2d}^{adapt}$", fontsize=6.5, color=C_MRL,
            ha="center", va="center", zorder=5)

    # ── KEY DIFFERENCES ANNOTATION ────────────────────────────
    ann_y = 4.15
    ax.text(cx, ann_y,
            "Key differences from v1:   ONE text encoder  ·  "
            "Exact 3D Gramian  ·  Training-free uncertainty  ·  "
            "No bridge / ExMCR / ProbVLM",
            fontsize=5.5, color=C_SUB, ha="center", fontstyle="italic")

    # ── LEGEND ────────────────────────────────────────────────
    legend_items = [
        (C_GEMINI, "Gemini"),
        (C_GRAM,   "Gramian"),
        (C_CALIB,  "Calibration"),
        (C_NEG,    "Contrastive"),
        (C_MRL,    "Matryoshka"),
        (C_OUT,    "Output"),
    ]
    ly = 3.70
    spacing = 1.4
    total_w = len(legend_items) * spacing
    lx_start = (W - total_w) / 2 + 0.1

    for i, (c, lab) in enumerate(legend_items):
        xo = lx_start + i * spacing
        p = FancyBboxPatch((xo, ly), 0.16, 0.12, boxstyle="round,pad=0.01",
                           facecolor=c, edgecolor=c, linewidth=0.4, zorder=3)
        ax.add_patch(p)
        ax.text(xo + 0.24, ly + 0.06, lab, fontsize=5.5, color=C_TEXT,
                va="center", zorder=4)

    # ── SAVE ──────────────────────────────────────────────────
    plt.tight_layout(pad=0.3)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_architecture_v2.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"v2 architecture diagram saved to {out}")


# ══════════════════════════════════════════════════════════════
#  DIAGRAM 2:  ENSEMBLE  (v1 + v2 Fusion)
# ══════════════════════════════════════════════════════════════

def draw_ensemble_architecture():
    W, H = 10.0, 14.0
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    cx = W / 2
    margin = 0.4
    full_w = W - 2 * margin

    # ── TITLE ─────────────────────────────────────────────────
    ax.text(cx, 13.60, "cMSCI Ensemble Architecture",
            ha="center", fontsize=14, fontweight="bold", color=C_TEXT,
            fontfamily="serif")
    ax.text(cx, 13.30,
            "Dual-Backbone Fusion:  CLIP+CLAP  +  Gemini Unified",
            ha="center", fontsize=8, color=C_SUB, fontstyle="italic")

    # ── INPUTS ────────────────────────────────────────────────
    iw, ih = 1.9, 0.38
    iy = 12.80
    i_centers = [1.5, 5.0, 8.5]
    for lbl, xc in zip(["Text", "Image", "Audio"], i_centers):
        _box(ax, xc - iw/2, iy, iw, ih, lbl, C_INPUT, C_TEXT, fontsize=9.5, lw=0.5)

    # ── STAGE 0: PARALLEL BACKBONES ───────────────────────────
    s0_top = 12.40
    s0_bot = 10.85
    _stage_label(ax, margin + 0.1, s0_top + 0.12, 0, "Parallel Embedding Backbones")

    # Left: v1 backbone (CLIP+CLAP)
    col_w_bb = 4.3
    col_gap = 0.6
    bb_x = [margin, margin + col_w_bb + col_gap]

    _stage_bg(ax, bb_x[0], s0_bot, col_w_bb, s0_top - s0_bot)
    _stage_bg(ax, bb_x[1], s0_bot, col_w_bb, s0_top - s0_bot)

    bb_cx = [bb_x[0] + col_w_bb/2, bb_x[1] + col_w_bb/2]

    # v1 encoders
    enc_h = 0.40
    enc_y = s0_bot + 0.10
    _box(ax, bb_x[0] + 0.10, enc_y + 0.50, 1.95, enc_h,
         "CLIP Text", C_CLIP, fontsize=6.5, sublabel="512-d", sub_fs=5.5)
    _box(ax, bb_x[0] + 2.20, enc_y + 0.50, 1.95, enc_h,
         "CLIP Image", C_CLIP, fontsize=6.5, sublabel="512-d", sub_fs=5.5)
    _box(ax, bb_x[0] + 0.10, enc_y, 1.95, enc_h,
         "CLAP Text", C_CLAP, fontsize=6.5, sublabel="512-d", sub_fs=5.5)
    _box(ax, bb_x[0] + 2.20, enc_y, 1.95, enc_h,
         "CLAP Audio", C_CLAP, fontsize=6.5, sublabel="512-d", sub_fs=5.5)

    ax.text(bb_cx[0], s0_top - 0.05, "Backbone 1: CLIP + CLAP", fontsize=7,
            color=C_V1, ha="center", fontweight="bold", zorder=5)
    ax.text(bb_cx[0], s0_bot + 1.15, "Dual spaces", fontsize=5.5,
            color=C_SUB, ha="center", fontstyle="italic", zorder=5)

    # v2 encoder
    _box(ax, bb_x[1] + 0.10, enc_y + 0.15, col_w_bb - 0.20, 0.60,
         "Gemini Embedding 2", C_GEMINI,
         fontsize=8, sublabel="3072-d unified space", sub_fs=6)

    ax.text(bb_cx[1], s0_top - 0.05, "Backbone 2: Gemini", fontsize=7,
            color=C_V2, ha="center", fontweight="bold", zorder=5)
    ax.text(bb_cx[1], s0_bot + 1.15, "Single unified space", fontsize=5.5,
            color=C_SUB, ha="center", fontstyle="italic", zorder=5)

    # Input → Backbones
    for ic in i_centers:
        _arr(ax, ic, iy, bb_cx[0], s0_top, C_V1, 0.7)
        _arr(ax, ic, iy, bb_cx[1], s0_top, C_V2, 0.7)

    # ── STAGE 1: PARALLEL PIPELINES ──────────────────────────
    s1_top = 10.45
    s1_bot = 7.20
    _stage_label(ax, margin + 0.1, s1_top + 0.12, 1, "Independent Calibration Pipelines")

    # v1 pipeline
    _stage_bg(ax, bb_x[0], s1_bot, col_w_bb, s1_top - s1_bot)
    # v2 pipeline
    _stage_bg(ax, bb_x[1], s1_bot, col_w_bb, s1_top - s1_bot)

    # v1 pipeline stages (stacked vertically)
    v1_boxes = [
        ("Gramian Volume (2D)", C_GRAM, "TI (CLIP) + TA (CLAP)"),
        ("Z-Score Calibration", C_CALIB, "CLIP/CLAP reference dists"),
        ("Contrastive Margin", C_NEG, r"$\alpha=7$, per-channel"),
        ("Ex-MCR Complementarity", C_EXMCR, r"CLAP$\to$CLIP, $w_{3d}=0.35$"),
        ("ProbVLM Adaptive", C_PROB, r"BayesCap, $\gamma=0.4$"),
    ]
    bw = col_w_bb - 0.20
    bh = 0.48
    by_start = s1_top - 0.15
    by_step = 0.60

    for i, (lbl, col, sub) in enumerate(v1_boxes):
        by = by_start - (i + 1) * by_step + 0.12
        _box(ax, bb_x[0] + 0.10, by, bw, bh, lbl, col,
             fontsize=6, sublabel=sub, sub_fs=5)
        if i > 0:
            _arr(ax, bb_cx[0], by + bh + 0.12, bb_cx[0], by + bh, C_ARR, 0.4)

    # v2 pipeline stages
    v2_boxes = [
        ("Gramian Volume (2D + 3D)", C_GRAM, "TI + TA  (IA=noise, excluded)"),
        ("Z-Score Calibration", C_CALIB, "Gemini reference dists"),
        ("3-Ch Contrastive", C_NEG, r"$\alpha=0$ (vanishes)"),
        ("Matryoshka Uncertainty", C_MRL, "768 / 1536 / 3072 consistency"),
        ("Adaptive Channel Weights", C_MRL, r"$\gamma_{mrl}=0.6$, $w_{ti}=0.90$"),
    ]

    for i, (lbl, col, sub) in enumerate(v2_boxes):
        by = by_start - (i + 1) * by_step + 0.12
        _box(ax, bb_x[1] + 0.10, by, bw, bh, lbl, col,
             fontsize=6, sublabel=sub, sub_fs=5)
        if i > 0:
            _arr(ax, bb_cx[1], by + bh + 0.12, bb_cx[1], by + bh, C_ARR, 0.4)

    # Backbone → pipeline
    _arr(ax, bb_cx[0], s0_bot, bb_cx[0], s1_top, C_V1, 0.8)
    _arr(ax, bb_cx[1], s0_bot, bb_cx[1], s1_top, C_V2, 0.8)

    # ── INTERMEDIATE SCORES ───────────────────────────────────
    mid_y = 6.55
    score_h = 0.55

    _box(ax, bb_x[0] + 0.30, mid_y, col_w_bb - 0.60, score_h,
         "cMSCI v1", C_V1, fontsize=9,
         sublabel=r"$\sigma(z_{2d}^{adapt} + \alpha m + w_{3d} z_{compl})$", sub_fs=6)
    _box(ax, bb_x[1] + 0.30, mid_y, col_w_bb - 0.60, score_h,
         "cMSCI v2", C_V2, fontsize=9,
         sublabel=r"$\sigma(w_{ti}^{f} z_{ti} + (1-w_{ti}^{f}) z_{ta})$", sub_fs=6)

    _arr(ax, bb_cx[0], s1_bot, bb_cx[0], mid_y + score_h, C_V1, 1.0)
    _arr(ax, bb_cx[1], s1_bot, bb_cx[1], mid_y + score_h, C_V2, 1.0)

    # Individual results
    ax.text(bb_cx[0], mid_y - 0.18, r"$\rho = 0.579$  ($p = 0.001$)",
            fontsize=6, color=C_V1, ha="center", fontstyle="italic")
    ax.text(bb_cx[1], mid_y - 0.18, r"$\rho = 0.558$  ($p = 0.001$)",
            fontsize=6, color=C_V2, ha="center", fontstyle="italic")

    # ── WHY ENSEMBLE WORKS ────────────────────────────────────
    why_y = 5.45
    why_h = 0.72
    _stage_bg(ax, margin, why_y, full_w, why_h)

    ax.text(cx, why_y + why_h - 0.15,
            "Why ensemble works:  errors are uncorrelated",
            fontsize=7.5, color=C_TEXT, ha="center", fontweight="bold", zorder=5)

    ax.text(cx - 2.5, why_y + 0.18,
            "v1: audio-dominant ($w_{ti}=0.30$)\n"
            "     better at audio-heavy samples",
            fontsize=5.5, color=C_V1, ha="center", zorder=5, family="monospace")
    ax.text(cx + 2.5, why_y + 0.18,
            "v2: image-dominant ($w_{ti}=0.90$)\n"
            "     better at image-heavy samples",
            fontsize=5.5, color=C_V2, ha="center", zorder=5, family="monospace")

    ax.text(cx, why_y + 0.18,
            r"$\leftrightarrow$",
            fontsize=14, color=C_SUB, ha="center", va="center", zorder=5)

    # ── ENSEMBLE FUSION ───────────────────────────────────────
    ens_y = 3.55
    ens_h = 1.60
    _stage_label(ax, margin + 0.1, ens_y + ens_h + 0.15, 2, "LOO-CV Optimized Fusion")
    _stage_bg(ax, margin, ens_y, full_w, ens_h)
    _box(ax, margin + 0.15, ens_y + 0.10, full_w - 0.30, ens_h - 0.20,
         "", C_ENS, fontsize=12, lw=1.5)

    ax.text(cx, ens_y + ens_h - 0.28, "Ensemble Score", fontsize=13, fontweight="bold",
            color="white", ha="center", va="center", zorder=4, fontfamily="serif")

    ax.text(cx, ens_y + ens_h - 0.62,
            r"$\mathrm{cMSCI}_{ens} = w_{v1} \cdot \mathrm{cMSCI}_{v1}"
            r" + (1 - w_{v1}) \cdot \mathrm{cMSCI}_{v2}$",
            fontsize=10, color="white", ha="center", va="center", zorder=4)

    ax.text(cx, ens_y + ens_h - 0.95,
            r"$w_{v1} = 0.4$   (30/30 LOO-CV folds unanimous)",
            fontsize=7.5, color="#FFD700", ha="center", va="center", zorder=4,
            fontweight="bold")

    ax.text(cx, ens_y + ens_h - 1.22,
            r"$\rho = 0.693$   ($p = 0.00002$)   LOO-CV $\rho = 0.693$   (zero overfitting)",
            fontsize=7, color="#F5C6C6", ha="center", va="center", zorder=4)

    # v1/v2 → ensemble
    _arr(ax, bb_cx[0], mid_y, cx - 1.5, ens_y + ens_h, C_V1, 1.2)
    _arr(ax, bb_cx[1], mid_y, cx + 1.5, ens_y + ens_h, C_V2, 1.2)

    # Weight labels on arrows
    ax.text(bb_cx[0] + 0.6, 5.9, r"$\times 0.4$", fontsize=8, color=C_V1,
            ha="center", va="center", zorder=5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.06", facecolor="white",
                      edgecolor=C_V1, linewidth=0.6, alpha=0.9))
    ax.text(bb_cx[1] - 0.6, 5.9, r"$\times 0.6$", fontsize=8, color=C_V2,
            ha="center", va="center", zorder=5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.06", facecolor="white",
                      edgecolor=C_V2, linewidth=0.6, alpha=0.9))

    # ── COMPARISON TABLE ──────────────────────────────────────
    tbl_y = 2.40
    ax.text(cx, tbl_y + 0.75, "Performance Comparison", fontsize=8,
            color=C_TEXT, ha="center", fontweight="bold")

    rows = [
        ("cMSCI v1 (CLIP+CLAP)", "0.579", "0.001", C_V1),
        ("cMSCI v2 (Gemini)", "0.558", "0.001", C_V2),
        ("Ensemble (0.4v1 + 0.6v2)", "0.693", "0.00002", C_ENS),
    ]

    # Header
    ax.text(cx - 2.5, tbl_y + 0.45, "Method", fontsize=6.5, color=C_SUB,
            ha="center", fontweight="bold")
    ax.text(cx + 1.2, tbl_y + 0.45, r"$\rho$", fontsize=6.5, color=C_SUB,
            ha="center", fontweight="bold")
    ax.text(cx + 2.8, tbl_y + 0.45, "p-value", fontsize=6.5, color=C_SUB,
            ha="center", fontweight="bold")

    ax.plot([cx - 4.2, cx + 3.8], [tbl_y + 0.35, tbl_y + 0.35],
            color=C_BORDER, lw=0.5, zorder=1)

    for i, (name, rho, p, col) in enumerate(rows):
        ry = tbl_y + 0.10 - i * 0.25
        ax.text(cx - 2.5, ry, name, fontsize=6.5, color=col,
                ha="center", fontweight="bold")
        ax.text(cx + 1.2, ry, rho, fontsize=6.5, color=col, ha="center")
        ax.text(cx + 2.8, ry, p, fontsize=6.5, color=col, ha="center")

    # Improvement annotation
    ax.text(cx, tbl_y - 0.55,
            r"Ensemble beats best single model by $+0.115$ in Spearman $\rho$",
            fontsize=6, color=C_ENS, ha="center", fontstyle="italic")

    # ── LEGEND ────────────────────────────────────────────────
    legend_items = [
        (C_V1,     "v1 (CLIP+CLAP)"),
        (C_V2,     "v2 (Gemini)"),
        (C_GRAM,   "Gramian"),
        (C_NEG,    "Contrastive"),
        (C_MRL,    "Matryoshka"),
        (C_ENS,    "Ensemble"),
    ]
    ly = 1.35
    spacing = 1.5
    total_w = len(legend_items) * spacing
    lx_start = (W - total_w) / 2 + 0.1

    for i, (c, lab) in enumerate(legend_items):
        xo = lx_start + i * spacing
        p = FancyBboxPatch((xo, ly), 0.16, 0.12, boxstyle="round,pad=0.01",
                           facecolor=c, edgecolor=c, linewidth=0.4, zorder=3)
        ax.add_patch(p)
        ax.text(xo + 0.24, ly + 0.06, lab, fontsize=5.5, color=C_TEXT,
                va="center", zorder=4)

    # ── SAVE ──────────────────────────────────────────────────
    plt.tight_layout(pad=0.3)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_architecture_ensemble.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Ensemble architecture diagram saved to {out}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    draw_v2_architecture()
    draw_ensemble_architecture()
