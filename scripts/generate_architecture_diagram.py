#!/usr/bin/env python3
"""
Generate a publication-quality cMSCI architecture diagram.

Clean block diagram with correct data flow. No curved skip arrows —
each stage box shows its input source via small annotations.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def draw_architecture():
    W, H = 10.0, 16.0
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Colors ────────────────────────────────────────────────
    C_CLIP   = "#3B7DD8"
    C_CLAP   = "#D4722C"
    C_STAGE  = "#F5F5F5"
    C_BORDER = "#C8C8C8"
    C_GRAM   = "#2D8B57"
    C_CALIB  = "#555555"
    C_NEG    = "#6B5B95"
    C_EXMCR  = "#1E8C5E"
    C_PROB   = "#7B4FA0"
    C_OUT    = "#B8352D"
    C_INPUT  = "#E0E0E0"
    C_TEXT   = "#1A1A1A"
    C_SUB    = "#777777"
    C_ARR    = "#444444"

    cx = W / 2
    margin = 0.4          # left/right margin
    full_w = W - 2 * margin  # full-width stage

    # ── Helpers ───────────────────────────────────────────────

    def stage_bg(x, y, w, h):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=C_STAGE, edgecolor=C_BORDER, linewidth=0.6, zorder=1)
        ax.add_patch(p)

    def stage_label(x, y, num, text):
        ax.text(x, y, f"Stage {num}:  {text}",
                fontsize=7, fontweight="bold", color=C_SUB, va="center", zorder=5)

    def box(x, y, w, h, label, color, text_color="white",
            fontsize=8.5, sublabel=None, sub_fs=7, lw=1.0):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04",
            facecolor=color, edgecolor=color, linewidth=lw, zorder=3)
        ax.add_patch(p)
        if sublabel:
            ax.text(x + w/2, y + h/2 + 0.11, label, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
            ax.text(x + w/2, y + h/2 - 0.11, sublabel, ha="center", va="center",
                    fontsize=sub_fs, color=text_color, zorder=4, alpha=0.85)
        else:
            ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)

    def arr(x1, y1, x2, y2, color=C_ARR, lw=1.0):
        a = FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", color=color,
            linewidth=lw, zorder=2, mutation_scale=10)
        ax.add_patch(a)

    def tag(x, y, text, color, fs=5.5):
        """Small source-tag annotation."""
        ax.text(x, y, text, fontsize=fs, color=color, ha="center", va="center",
                zorder=5, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.06", facecolor="white",
                          edgecolor=color, alpha=0.85, linewidth=0.4))

    # ══════════════════════════════════════════════════════════
    # TITLE
    # ══════════════════════════════════════════════════════════
    ax.text(cx, 15.65, "cMSCI Pipeline Architecture",
            ha="center", fontsize=14, fontweight="bold", color=C_TEXT,
            fontfamily="serif")
    ax.text(cx, 15.35,
            "Calibrated Multimodal Semantic Coherence Index",
            ha="center", fontsize=8, color=C_SUB, fontstyle="italic")

    # ══════════════════════════════════════════════════════════
    # INPUTS (y ~ 14.8)
    # ══════════════════════════════════════════════════════════
    iw, ih = 1.9, 0.38
    iy = 14.85
    i_centers = [1.5, 5.0, 8.5]
    for lbl, xc in zip(["Text", "Image", "Audio"], i_centers):
        box(xc - iw/2, iy, iw, ih, lbl, C_INPUT, C_TEXT, fontsize=9.5, lw=0.5)

    # ══════════════════════════════════════════════════════════
    # STAGE 0: ENCODERS
    # ══════════════════════════════════════════════════════════
    s0y = 13.35
    stage_label(margin + 0.1, 14.45, 0, "Frozen Embedding Extraction")
    stage_bg(margin, s0y, full_w, 1.0)

    ew, eh = 2.05, 0.45
    ey = s0y + 0.10
    e_x = [0.55, 2.75, 5.25, 7.45]
    e_labels = ["CLIP Text Enc.", "CLIP Image Enc.", "CLAP Text Enc.", "CLAP Audio Enc."]
    e_sub = ["ViT-B/32, 512-d"] * 2 + ["HTSAT, 512-d"] * 2
    e_col = [C_CLIP, C_CLIP, C_CLAP, C_CLAP]
    e_cx = []
    for x, l, s, c in zip(e_x, e_labels, e_sub, e_col):
        box(x, ey, ew, eh, l, c, fontsize=7, sublabel=s, sub_fs=6)
        e_cx.append(x + ew / 2)

    # Space separator
    ax.plot([5.0, 5.0], [s0y + 0.03, s0y + 0.97], color="#D0D0D0",
            lw=0.6, ls=":", zorder=1)
    ax.text(2.2, s0y + 0.85, "CLIP space (512-d)", fontsize=6, color=C_CLIP,
            ha="center", fontweight="bold")
    ax.text(7.8, s0y + 0.85, "CLAP space (512-d)", fontsize=6, color=C_CLAP,
            ha="center", fontweight="bold")

    # Input → Encoder
    arr(i_centers[0], iy, e_cx[0], ey + eh, C_CLIP, 0.8)
    arr(i_centers[0], iy, e_cx[2], ey + eh, C_CLAP, 0.8)
    arr(i_centers[1], iy, e_cx[1], ey + eh, C_CLIP, 0.8)
    arr(i_centers[2], iy, e_cx[3], ey + eh, C_CLAP, 0.8)

    # ══════════════════════════════════════════════════════════
    # STAGE 1: GRAMIAN VOLUME
    # ══════════════════════════════════════════════════════════
    s1y = 11.95
    stage_label(margin + 0.1, 13.05, 1, "Gramian Volume Geometry")
    stage_bg(margin, s1y, full_w, 1.0)

    gw, gh = 4.2, 0.48
    gy = s1y + 0.10
    box(margin + 0.15, gy, gw, gh, "Gramian Volume (TI)", C_GRAM, fontsize=7.5,
        sublabel=r"$V_{ti} = \sqrt{1 - \cos^2\theta_{ti}}$", sub_fs=6.5)
    box(W - margin - 0.15 - gw, gy, gw, gh, "Gramian Volume (TA)", C_GRAM, fontsize=7.5,
        sublabel=r"$V_{ta} = \sqrt{1 - \cos^2\theta_{ta}}$", sub_fs=6.5)

    g_cx_l = margin + 0.15 + gw / 2
    g_cx_r = W - margin - 0.15 - gw / 2

    ax.text(cx, s1y + 0.82,
            r"Coherence:  $c = 1 - V$    (0 = orthogonal,  1 = aligned)",
            fontsize=6, color=C_GRAM, ha="center", fontstyle="italic", zorder=5)

    # Stage 0 → Stage 1
    arr(e_cx[0], ey, g_cx_l - 0.8, gy + gh, C_CLIP, 0.6)
    arr(e_cx[1], ey, g_cx_l + 0.8, gy + gh, C_CLIP, 0.6)
    arr(e_cx[2], ey, g_cx_r - 0.8, gy + gh, C_CLAP, 0.6)
    arr(e_cx[3], ey, g_cx_r + 0.8, gy + gh, C_CLAP, 0.6)

    # ══════════════════════════════════════════════════════════
    # STAGE 2: Z-CALIBRATION
    # ══════════════════════════════════════════════════════════
    s2y = 10.60
    stage_label(margin + 0.1, 11.65, 2, "Z-Score Calibration")
    stage_bg(margin, s2y, full_w, 0.95)

    zw, zh = 4.2, 0.45
    zy = s2y + 0.10
    box(margin + 0.15, zy, zw, zh, r"$z_{ti}$", C_CALIB, fontsize=8.5,
        sublabel=r"$(c_{ti} - \mu_{ti})\, /\, \sigma_{ti}$", sub_fs=6.5)
    box(W - margin - 0.15 - zw, zy, zw, zh, r"$z_{ta}$", C_CALIB, fontsize=8.5,
        sublabel=r"$(c_{ta} - \mu_{ta})\, /\, \sigma_{ta}$", sub_fs=6.5)

    z_cx_l = margin + 0.15 + zw / 2
    z_cx_r = W - margin - 0.15 - zw / 2

    # Stage 1 → Stage 2
    arr(g_cx_l, gy, z_cx_l, zy + zh, C_ARR, 0.6)
    arr(g_cx_r, gy, z_cx_r, zy + zh, C_ARR, 0.6)

    # ══════════════════════════════════════════════════════════
    # STAGES 3, 4, 5 — THREE PARALLEL BRANCHES
    # All at the same vertical level, each labeled with source
    # ══════════════════════════════════════════════════════════

    br_top = 10.20
    br_bot = 6.80
    br_h = br_top - br_bot
    col_w = 2.85
    col_gap = 0.25
    col_x = [margin,
             margin + col_w + col_gap,
             margin + 2 * (col_w + col_gap)]

    # ── STAGE 3: CONTRASTIVE MARGIN (left) ─────────────────
    stage_label(col_x[0] + 0.1, br_top + 0.12, 3, "Contrastive Margin")
    stage_bg(col_x[0], br_bot, col_w, br_h)

    s3_cx = col_x[0] + col_w / 2

    # Source annotation
    tag(s3_cx, 9.90, "from Stage 1:  V_ti, V_ta", C_GRAM, 5)

    box(col_x[0] + 0.10, 9.35, col_w - 0.20, 0.42, "Hard Negative Mining", C_NEG,
        fontsize=7, sublabel="k = 5 per channel", sub_fs=5.5)

    box(col_x[0] + 0.10, 8.50, col_w - 0.20, 0.65, "", C_NEG, fontsize=7)
    ax.text(s3_cx, 8.97, "Per-Channel Margins", fontsize=6.5,
            fontweight="bold", color="white", ha="center", va="center", zorder=4)
    ax.text(s3_cx, 8.76,
            r"$m_{ti} = \overline{V}_{ti}^{neg} - V_{ti}^{*}$",
            fontsize=6, color="white", ha="center", va="center", zorder=4)
    ax.text(s3_cx, 8.57,
            r"$m_{ta} = \overline{V}_{ta}^{neg} - V_{ta}^{*}$",
            fontsize=6, color="white", ha="center", va="center", zorder=4)

    box(col_x[0] + 0.10, 7.55, col_w - 0.20, 0.48, "", C_NEG, fontsize=7)
    ax.text(s3_cx, 7.87, "Combined", fontsize=6.5,
            fontweight="bold", color="white", ha="center", va="center", zorder=4)
    ax.text(s3_cx, 7.65,
            r"$m = w_{ti} m_{ti} + (1\!-\!w_{ti}) m_{ta}$",
            fontsize=5.5, color="white", ha="center", va="center", zorder=4)

    arr(s3_cx, 9.35, s3_cx, 9.15, C_NEG, 0.5)
    arr(s3_cx, 8.50, s3_cx, 8.03, C_NEG, 0.5)
    # Output label
    tag(s3_cx, 7.18, r"output:  $m$", C_NEG, 5.5)

    # ── STAGE 4: EX-MCR (center) ──────────────────────────
    stage_label(col_x[1] + 0.1, br_top + 0.12, 4, "Cross-Space Compl.")
    stage_bg(col_x[1], br_bot, col_w, br_h)

    s4_cx = col_x[1] + col_w / 2

    # Source annotation
    tag(s4_cx, 9.90, "from Stage 0:  e_img, e_aud", C_CLIP, 5)

    box(col_x[1] + 0.10, 9.30, col_w - 0.20, 0.45, "Ex-MCR Projection", C_EXMCR,
        fontsize=7, sublabel=r"CLAP$\to$CLIP (MLP, 525K)", sub_fs=5.5)

    box(col_x[1] + 0.10, 8.50, col_w - 0.20, 0.52, "Gramian Dispersion", C_GRAM,
        fontsize=7, sublabel=r"$V_{ia} = |\sin\theta(\mathbf{e}_{img}, \mathbf{e}_{aud}^{proj})|$",
        sub_fs=5.5)

    box(col_x[1] + 0.10, 7.20, col_w - 0.20, 0.88, "", C_EXMCR, fontsize=7)
    ax.text(s4_cx, 7.95, "Complementarity", fontsize=6.5,
            fontweight="bold", color="white", ha="center", va="center", zorder=4)
    ax.text(s4_cx, 7.76,
            r"$\mathrm{coh}_{ia} = 1 - V_{ia}$",
            fontsize=5.5, color="white", ha="center", va="center", zorder=4)
    ax.text(s4_cx, 7.58,
            r"$z_{\mathrm{coh}_{ia}} = (\mathrm{coh}_{ia} - \mu_{ia})\, /\, \sigma_{ia}$",
            fontsize=5.5, color="white", ha="center", va="center", zorder=4)
    ax.text(s4_cx, 7.40,
            r"$z_{compl} = - z_{\mathrm{coh}_{ia}}$   ← sign flip",
            fontsize=6, color="#FFD700", ha="center", va="center", zorder=4)
    ax.text(s4_cx, 7.22, "sign flip: dispersion = complementarity",
            fontsize=4.5, color="#A0D8B8", ha="center", fontstyle="italic", zorder=5)

    arr(s4_cx, 9.30, s4_cx, 9.02, C_EXMCR, 0.5)
    arr(s4_cx, 8.50, s4_cx, 8.07, C_EXMCR, 0.5)
    # Output label
    tag(s4_cx, 7.18, r"output:  $z_{compl}$", C_EXMCR, 5.5)

    # ── STAGE 5: ProbVLM (right) ──────────────────────────
    stage_label(col_x[2] + 0.1, br_top + 0.12, 5, "Adaptive Weighting")
    stage_bg(col_x[2], br_bot, col_w, br_h)

    s5_cx = col_x[2] + col_w / 2

    # Source annotation
    tag(s5_cx, 9.90, "from Stages 0+2:  embs, z_ti, z_ta", C_PROB, 5)

    box(col_x[2] + 0.10, 9.30, col_w - 0.20, 0.42, "CLIP Adapter", C_PROB,
        fontsize=7, sublabel=r"$u_{ti} = \mathrm{mean}(\sigma_{clip})$  592K", sub_fs=5.5)

    box(col_x[2] + 0.10, 8.60, col_w - 0.20, 0.42, "CLAP Adapter", C_PROB,
        fontsize=7, sublabel=r"$u_{ta} = \mathrm{mean}(\sigma_{clap})$  592K", sub_fs=5.5)

    box(col_x[2] + 0.10, 7.55, col_w - 0.20, 0.72, "", C_PROB, fontsize=7)
    ax.text(s5_cx, 8.10, "Adaptive Weight", fontsize=6.5,
            fontweight="bold", color="white", ha="center", va="center", zorder=4)
    ax.text(s5_cx, 7.88,
            r"$w_{ti}^{f} = (1\!-\!\gamma) w_{ti}^{b}"
            r" + \gamma \!\cdot\! \frac{1/u_{ti}}{1/u_{ti}\!+\!1/u_{ta}}$",
            fontsize=5.5, color="white", ha="center", va="center", zorder=4)
    ax.text(s5_cx, 7.68,
            r"$z_{2d}^{adapt} = w_{ti}^{f} z_{ti} + (1\!-\!w_{ti}^{f}) z_{ta}$",
            fontsize=5.5, color="#D4C4E8", ha="center", va="center",
            zorder=4, fontstyle="italic")

    # Both adapters feed independently into Adaptive Weight
    arr(s5_cx - 0.35, 9.30, s5_cx - 0.35, 8.27, C_PROB, 0.5)  # CLIP Adapter → Weight
    arr(s5_cx + 0.35, 8.60, s5_cx + 0.35, 8.27, C_PROB, 0.5)  # CLAP Adapter → Weight
    # Small labels on adapter arrows
    ax.text(s5_cx - 0.55, 8.45, r"$u_{ti}$", fontsize=5, color=C_PROB,
            ha="center", va="center", zorder=5)
    ax.text(s5_cx + 0.55, 8.45, r"$u_{ta}$", fontsize=5, color=C_PROB,
            ha="center", va="center", zorder=5)
    # Output label
    tag(s5_cx, 7.18, r"output:  $z_{2d}^{adapt}$, $w_{ti}^{final}$", C_PROB, 5.5)

    # ══════════════════════════════════════════════════════════
    # ARROWS: correct data flow
    # Stage 1 → Stage 3 (volumes for margin computation)
    # Stage 0 → Stage 4 (embeddings for ExMCR projection)
    # Stage 2 → Stage 5 (z_ti, z_ta for adaptive reweighting)
    # ══════════════════════════════════════════════════════════
    arr(g_cx_l, gy, s3_cx, br_top, C_GRAM, 0.7)          # Stage 1 → Stage 3 (volumes)
    arr(e_cx[1], ey, s4_cx - 0.3, br_top, C_CLIP, 0.7)  # Stage 0 CLIP img → Stage 4
    arr(e_cx[3], ey, s4_cx + 0.3, br_top, C_CLAP, 0.7)  # Stage 0 CLAP aud → Stage 4
    arr(z_cx_r, zy, s5_cx, br_top, C_CALIB, 0.7)        # Stage 2 → Stage 5 (z-scores)

    # ══════════════════════════════════════════════════════════
    # OUTPUT: cMSCI (y ~ 4.5)
    # ══════════════════════════════════════════════════════════
    out_y = 4.65
    out_h = 1.80
    stage_bg(margin, out_y, full_w, out_h)
    box(margin + 0.1, out_y + 0.10, full_w - 0.2, out_h - 0.20, "", C_OUT,
        fontsize=12, lw=1.5)

    ax.text(cx, out_y + out_h - 0.35, "cMSCI Score", fontsize=12, fontweight="bold",
            color="white", ha="center", va="center", zorder=4,
            fontfamily="serif")

    ax.text(cx, out_y + out_h - 0.75,
            r"$\mathrm{cMSCI} = \sigma\!\left("
            r"z_{2d}^{adapt}"
            r"\ +\ \alpha \cdot m"
            r"\ +\ w_{3d} \cdot z_{compl}"
            r"\right)$",
            fontsize=9.5, color="white", ha="center", va="center", zorder=4)

    ax.text(cx, out_y + out_h - 1.10,
            r"$\sigma(x) = \frac{1}{1+e^{-x}} \in [0,1]$"
            r"$\qquad$"
            r"$\alpha\!=\!7$   $w_{ti}^{b}\!=\!0.30$   $w_{3d}\!=\!0.35$   $\gamma\!=\!0.40$",
            fontsize=6.5, color="#F5C6C6", ha="center", va="center", zorder=4)

    ax.text(cx, out_y + out_h - 1.40,
            r"Safety:  $\gamma\!=\!0$ recovers Variant E;  "
            r"$w_{3d}\!=\!0$ recovers D;  "
            r"$\alpha\!=\!0$ recovers C",
            fontsize=5.5, color="#F5C6C6", ha="center", va="center",
            zorder=4, fontstyle="italic")

    # Branches → Output
    arr(s3_cx, br_bot, cx - 1.8, out_y + out_h, C_NEG, 1.0)
    arr(s4_cx, br_bot, cx, out_y + out_h, C_EXMCR, 1.0)
    arr(s5_cx, br_bot, cx + 1.8, out_y + out_h, C_PROB, 1.0)

    # Arrow labels
    ax.text(s3_cx - 0.6, 6.90, r"$\alpha \!\cdot\! m$", fontsize=6.5, color=C_NEG,
            ha="center", va="center", zorder=5)
    ax.text(s4_cx, 6.90, r"$w_{3d} \!\cdot\! z_{compl}$", fontsize=6.5, color=C_EXMCR,
            ha="center", va="center", zorder=5)
    ax.text(s5_cx + 0.6, 6.90, r"$z_{2d}^{adapt}$", fontsize=6.5, color=C_PROB,
            ha="center", va="center", zorder=5)

    # ══════════════════════════════════════════════════════════
    # LEGEND
    # ══════════════════════════════════════════════════════════
    legend_items = [
        (C_CLIP,  "CLIP"),
        (C_CLAP,  "CLAP"),
        (C_GRAM,  "Gramian"),
        (C_CALIB, "Calibration"),
        (C_NEG,   "Contrastive"),
        (C_EXMCR, "Ex-MCR"),
        (C_PROB,  "ProbVLM"),
        (C_OUT,   "Output"),
    ]
    ly = 4.15
    spacing = 1.1
    total_w = len(legend_items) * spacing
    lx_start = (W - total_w) / 2 + 0.1

    for i, (c, lab) in enumerate(legend_items):
        xo = lx_start + i * spacing
        p = FancyBboxPatch((xo, ly), 0.16, 0.12, boxstyle="round,pad=0.01",
                           facecolor=c, edgecolor=c, linewidth=0.4, zorder=3)
        ax.add_patch(p)
        ax.text(xo + 0.24, ly + 0.06, lab, fontsize=5.5, color=C_TEXT,
                va="center", zorder=4)

    # ══════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════
    plt.tight_layout(pad=0.3)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_architecture.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Architecture diagram saved to {out}")


if __name__ == "__main__":
    draw_architecture()
