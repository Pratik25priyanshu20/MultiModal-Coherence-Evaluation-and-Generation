#!/usr/bin/env python3
"""
Generate publication-quality figures + statistical refinements for the paper.

Outputs:
  figures/fig1_rq1_raincloud_f.pdf          — Raincloud plot: MSCI by perturbation condition
  figures/fig2_rq1_paired_slopes_f.pdf      — Paired slope (spaghetti) plot: per-prompt trajectories
  figures/fig3_rq2_estimation_f.pdf         — Gardner-Altman estimation plot: planning modes vs direct
  figures/fig4_forest_plot_f.pdf            — Forest plot: all effect sizes + CIs at a glance
  figures/fig5_rq1_channel_decomposition_f.pdf — Stacked bar: text-image vs text-audio channel breakdown
  figures/fig6_rq1_domain_heatmap_f.pdf     — Heatmap: MSCI by prompt domain × condition
  figures/fig7_rq2_power_curve_f.pdf        — Power curve: detectable effect sizes at N=30
  figures/statistical_supplement.txt      — Normality tests, Wilcoxon backup, Holm-Bonferroni RQ2
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, wilcoxon, nct, t as t_dist

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Palette
PAL_RQ1 = {"Baseline": "#2ecc71", "Wrong Image": "#e74c3c", "Wrong Audio": "#3498db"}
PAL_RQ2 = {"Direct": "#2c3e50", "Extended Prompt": "#8e44ad", "Council": "#e67e22", "Planner": "#16a085"}


# ── Load data ────────────────────────────────────────────────
def load_rq1(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for r in data["results"]:
        rows.append({
            "prompt_id": r["prompt_id"],
            "domain": r["domain"],
            "seed": r["seed"],
            "condition": r["condition"],
            "msci": r["msci"],
            "st_i": r["st_i"],
            "st_a": r["st_a"],
        })
    return pd.DataFrame(rows)


def load_rq2(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for r in data["results"]:
        rows.append({
            "prompt_id": r["prompt_id"],
            "domain": r["domain"],
            "seed": r["seed"],
            "mode": r["mode"],
            "msci": r["msci"],
            "st_i": r["st_i"],
            "st_a": r["st_a"],
        })
    return pd.DataFrame(rows)


rq1_skip = load_rq1(ROOT / "runs/rq1/rq1_results.json")
rq1_full = load_rq1(ROOT / "runs/rq1_full/rq1_results.json")
rq2 = load_rq2(ROOT / "runs/rq2/rq2_results.json")

# Average over seeds for per-prompt means
rq1_skip_agg = rq1_skip.groupby(["prompt_id", "domain", "condition"]).mean(numeric_only=True).reset_index()
rq1_full_agg = rq1_full.groupby(["prompt_id", "domain", "condition"]).mean(numeric_only=True).reset_index()
rq2_agg = rq2.groupby(["prompt_id", "domain", "mode"]).mean(numeric_only=True).reset_index()


# ═══════════════════════════════════════════════════════════
# STATISTICAL SUPPLEMENT
# ═══════════════════════════════════════════════════════════
sup_lines = []
sup_lines.append("=" * 70)
sup_lines.append("STATISTICAL SUPPLEMENT")
sup_lines.append("=" * 70)

# ── Normality tests ──────────────────────────────────────
sup_lines.append("\n1. NORMALITY TESTS (Shapiro-Wilk on paired differences)")
sup_lines.append("-" * 50)

def check_normality(df_agg, cond_col, baseline_val, perturb_val, label):
    base = df_agg[df_agg[cond_col] == baseline_val].set_index("prompt_id")["msci"]
    pert = df_agg[df_agg[cond_col] == perturb_val].set_index("prompt_id")["msci"]
    diff = (base - pert).dropna()
    w, p = shapiro(diff)
    normal = "Yes" if p > 0.05 else "No"
    sup_lines.append(f"  {label}: W = {w:.4f}, p = {p:.4f} → Normal: {normal}")
    return diff

# RQ1 skip-text
sup_lines.append("\nRQ1 Skip-text:")
d1 = check_normality(rq1_skip_agg, "condition", "baseline", "wrong_image", "baseline vs wrong_image")
d2 = check_normality(rq1_skip_agg, "condition", "baseline", "wrong_audio", "baseline vs wrong_audio")

sup_lines.append("\nRQ1 Full-pipeline:")
d3 = check_normality(rq1_full_agg, "condition", "baseline", "wrong_image", "baseline vs wrong_image")
d4 = check_normality(rq1_full_agg, "condition", "baseline", "wrong_audio", "baseline vs wrong_audio")

sup_lines.append("\nRQ2 (vs direct):")
for mode in ["council", "extended_prompt", "planner"]:
    direct = rq2_agg[rq2_agg["mode"] == "direct"].set_index("prompt_id")["msci"]
    other = rq2_agg[rq2_agg["mode"] == mode].set_index("prompt_id")["msci"]
    diff = (other - direct).dropna()
    w, p = shapiro(diff)
    normal = "Yes" if p > 0.05 else "No"
    sup_lines.append(f"  {mode} vs direct: W = {w:.4f}, p = {p:.4f} → Normal: {normal}")

# ── Wilcoxon signed-rank (non-parametric backup) ────────
sup_lines.append("\n2. WILCOXON SIGNED-RANK TESTS (non-parametric robustness check)")
sup_lines.append("-" * 50)

def wilcoxon_test(df_agg, cond_col, baseline_val, perturb_val, label):
    base = df_agg[df_agg[cond_col] == baseline_val].set_index("prompt_id")["msci"]
    pert = df_agg[df_agg[cond_col] == perturb_val].set_index("prompt_id")["msci"]
    diff = (base - pert).dropna()
    stat, p = wilcoxon(diff)
    n = len(diff)
    # rank-biserial correlation as effect size
    r = 1 - (2 * stat) / (n * (n + 1) / 2)
    sup_lines.append(f"  {label}: W = {stat:.1f}, p = {p:.2e}, r_rb = {r:.3f}, n = {n}")

sup_lines.append("\nRQ1 Skip-text:")
wilcoxon_test(rq1_skip_agg, "condition", "baseline", "wrong_image", "baseline vs wrong_image")
wilcoxon_test(rq1_skip_agg, "condition", "baseline", "wrong_audio", "baseline vs wrong_audio")

sup_lines.append("\nRQ1 Full-pipeline:")
wilcoxon_test(rq1_full_agg, "condition", "baseline", "wrong_image", "baseline vs wrong_image")
wilcoxon_test(rq1_full_agg, "condition", "baseline", "wrong_audio", "baseline vs wrong_audio")

sup_lines.append("\nRQ2:")
for mode in ["council", "extended_prompt", "planner"]:
    direct = rq2_agg[rq2_agg["mode"] == "direct"].set_index("prompt_id")["msci"]
    other = rq2_agg[rq2_agg["mode"] == mode].set_index("prompt_id")["msci"]
    diff = (direct - other).dropna()
    stat, p = wilcoxon(diff)
    n = len(diff)
    r = 1 - (2 * stat) / (n * (n + 1) / 2)
    sup_lines.append(f"  direct vs {mode}: W = {stat:.1f}, p = {p:.2e}, r_rb = {r:.3f}, n = {n}")

# ── Holm-Bonferroni for RQ2 ─────────────────────────────
sup_lines.append("\n3. HOLM-BONFERRONI CORRECTION FOR RQ2")
sup_lines.append("-" * 50)

rq2_pvals = []
for mode in ["council", "extended_prompt", "planner"]:
    direct = rq2_agg[rq2_agg["mode"] == "direct"].set_index("prompt_id")["msci"]
    other = rq2_agg[rq2_agg["mode"] == mode].set_index("prompt_id")["msci"]
    diff = (other - direct).dropna()
    t_stat, p = stats.ttest_rel(other.reindex(diff.index), direct.reindex(diff.index))
    rq2_pvals.append((f"{mode} vs direct", p, t_stat))

# Sort by p-value
m = len(rq2_pvals)
sorted_pvals = sorted(rq2_pvals, key=lambda x: x[1])
max_adj = 0
for rank, (label, p, t_s) in enumerate(sorted_pvals):
    adj = p * (m - rank)
    adj = max(adj, max_adj)
    adj = min(adj, 1.0)
    max_adj = adj
    sig = "significant" if adj < 0.05 else "not significant"
    sup_lines.append(f"  {label}: raw p = {p:.4f}, Holm-adjusted p = {adj:.4f} → {sig}")

# Write supplement
sup_text = "\n".join(sup_lines)
(FIG_DIR / "statistical_supplement.txt").write_text(sup_text)
print(sup_text)
print(f"\n→ Saved: {FIG_DIR / 'statistical_supplement.txt'}")


# ═══════════════════════════════════════════════════════════
# FIGURE 1: Raincloud Plot — RQ1 MSCI by Condition
# ═══════════════════════════════════════════════════════════
print("\n[Fig 1] Raincloud plot...")

# Use skip-text (primary analysis)
plot_df = rq1_skip_agg.copy()
cond_map = {"baseline": "Baseline", "wrong_image": "Wrong Image", "wrong_audio": "Wrong Audio"}
plot_df["Condition"] = plot_df["condition"].map(cond_map)
plot_df["Condition"] = pd.Categorical(plot_df["Condition"], categories=["Baseline", "Wrong Image", "Wrong Audio"], ordered=True)

fig, ax = plt.subplots(figsize=(8, 4.5))

conditions = ["Baseline", "Wrong Image", "Wrong Audio"]
colors = [PAL_RQ1[c] for c in conditions]

for i, cond in enumerate(conditions):
    data = plot_df[plot_df["Condition"] == cond]["msci"].values

    # Violin (half) — right side
    parts = ax.violinplot([data], positions=[i], showmeans=False, showmedians=False, showextrema=False, vert=True, widths=0.6)
    for pc in parts["bodies"]:
        # Clip to right half
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.3)
        pc.set_edgecolor("none")

    # Box plot — narrow, left side
    bp = ax.boxplot([data], positions=[i - 0.12], widths=0.15, vert=True, patch_artist=True,
                    showfliers=False, zorder=3)
    bp["boxes"][0].set_facecolor(colors[i])
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][0].set_edgecolor("black")
    bp["boxes"][0].set_linewidth(0.8)
    bp["medians"][0].set_color("black")
    bp["medians"][0].set_linewidth(1.5)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(0.8)

    # Individual points — jittered, left of box
    jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(data))
    ax.scatter(np.full(len(data), i - 0.28) + jitter, data,
               c=colors[i], s=18, alpha=0.6, edgecolors="white", linewidths=0.3, zorder=4)

    # Mean marker
    ax.scatter([i - 0.12], [np.mean(data)], c="white", s=40, marker="D",
               edgecolors="black", linewidths=1.0, zorder=5)

ax.set_xticks(range(3))
ax.set_xticklabels(conditions)
ax.set_ylabel("MSCI")
ax.set_title("RQ1: MSCI Distribution by Perturbation Condition (Skip-Text)")

# Add significance brackets
def add_bracket(ax, x1, x2, y, p_str, h=0.008):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, c="black")
    ax.text((x1+x2)/2, y+h+0.003, p_str, ha="center", va="bottom", fontsize=9)

ymax = plot_df["msci"].max()
add_bracket(ax, 0, 1, ymax + 0.02, "***\nd = 2.27", h=0.008)
add_bracket(ax, 0, 2, ymax + 0.08, "***\nd = 3.64", h=0.008)

ax.set_ylim(0, ymax + 0.16)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_rq1_raincloud_f.pdf")
fig.savefig(FIG_DIR / "fig1_rq1_raincloud_f.png")
plt.close(fig)
print(f"  → Saved: fig1_rq1_raincloud_f.pdf/png")


# ═══════════════════════════════════════════════════════════
# FIGURE 2: Paired Slope Plot — Per-Prompt Trajectories
# ═══════════════════════════════════════════════════════════
print("[Fig 2] Paired slope plot...")

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

for ax_idx, (perturb, perturb_label, d_val) in enumerate([
    ("wrong_image", "Wrong Image", "d = 2.27"),
    ("wrong_audio", "Wrong Audio", "d = 3.64"),
]):
    ax = axes[ax_idx]
    base = rq1_skip_agg[rq1_skip_agg["condition"] == "baseline"].set_index("prompt_id")
    pert = rq1_skip_agg[rq1_skip_agg["condition"] == perturb].set_index("prompt_id")

    common = base.index.intersection(pert.index)

    for pid in common:
        y_base = base.loc[pid, "msci"]
        y_pert = pert.loc[pid, "msci"]
        color = "#e74c3c" if y_pert < y_base else "#2ecc71"
        alpha = 0.4
        ax.plot([0, 1], [y_base, y_pert], c=color, alpha=alpha, linewidth=0.8, zorder=2)

    # Means
    mean_base = base.loc[common, "msci"].mean()
    mean_pert = pert.loc[common, "msci"].mean()
    ax.plot([0, 1], [mean_base, mean_pert], c="black", linewidth=2.5, zorder=5, label="Mean")
    ax.scatter([0, 1], [mean_base, mean_pert], c="black", s=60, zorder=6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", perturb_label])
    ax.set_title(f"Baseline → {perturb_label}\n({d_val}, p < .001)")
    ax.set_xlim(-0.3, 1.3)

    # Percentage of prompts that decreased
    n_decreased = sum(1 for pid in common if pert.loc[pid, "msci"] < base.loc[pid, "msci"])
    pct = n_decreased / len(common) * 100
    ax.text(0.5, 0.02, f"{pct:.0f}% of prompts decreased",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#555")

axes[0].set_ylabel("MSCI")
fig.suptitle("RQ1: Per-Prompt MSCI Change Under Perturbation", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_rq1_paired_slopes_f.pdf")
fig.savefig(FIG_DIR / "fig2_rq1_paired_slopes_f.png")
plt.close(fig)
print(f"  → Saved: fig2_rq1_paired_slopes_f.pdf/png")


# ═══════════════════════════════════════════════════════════
# FIGURE 3: Gardner-Altman Estimation Plot — RQ2
# ═══════════════════════════════════════════════════════════
print("[Fig 3] Gardner-Altman estimation plot...")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

direct = rq2_agg[rq2_agg["mode"] == "direct"].set_index("prompt_id")["msci"]

for ax_idx, (mode, color, label) in enumerate([
    ("council", PAL_RQ2["Council"], "Council"),
    ("extended_prompt", PAL_RQ2["Extended Prompt"], "Extended Prompt"),
    ("planner", PAL_RQ2["Planner"], "Planner"),
]):
    ax = axes[ax_idx]
    other = rq2_agg[rq2_agg["mode"] == mode].set_index("prompt_id")["msci"]
    common = direct.index.intersection(other.index)
    diff = other.loc[common] - direct.loc[common]

    # Left half: paired observations
    ax_left = ax
    for pid in common:
        ax_left.plot([0, 1], [direct.loc[pid], other.loc[pid]], c="#bdc3c7", alpha=0.4, linewidth=0.5)

    # Swarm-like scatter
    jit_d = np.random.default_rng(42).uniform(-0.08, 0.08, len(common))
    jit_o = np.random.default_rng(99).uniform(-0.08, 0.08, len(common))
    ax_left.scatter(np.zeros(len(common)) + jit_d, direct.loc[common].values,
                    c=PAL_RQ2["Direct"], s=20, alpha=0.6, edgecolors="white", linewidths=0.3, zorder=3)
    ax_left.scatter(np.ones(len(common)) + jit_o, other.loc[common].values,
                    c=color, s=20, alpha=0.6, edgecolors="white", linewidths=0.3, zorder=3)

    # Means
    ax_left.scatter([0], [direct.loc[common].mean()], c=PAL_RQ2["Direct"], s=80, marker="D",
                    edgecolors="black", linewidths=1.0, zorder=5)
    ax_left.scatter([1], [other.loc[common].mean()], c=color, s=80, marker="D",
                    edgecolors="black", linewidths=1.0, zorder=5)

    ax_left.set_xticks([0, 1])
    ax_left.set_xticklabels(["Direct", label], fontsize=9)
    ax_left.set_title(f"{label} vs Direct", fontsize=11)

    # Right axis: difference distribution
    ax_right = ax_left.twinx()
    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_means = [diff.sample(n=len(diff), replace=True, random_state=int(rng.integers(1e8))).mean() for _ in range(5000)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    mean_diff = diff.mean()

    # Plot CI as a vertical bar on the right
    ax_right.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_right.errorbar(1.6, mean_diff, yerr=[[mean_diff - ci_lo], [ci_hi - mean_diff]],
                      fmt="o", markersize=8, color=color, capsize=5, capthick=1.5, linewidth=1.5, zorder=5)
    ax_right.set_ylabel("Δ MSCI" if ax_idx == 2 else "", fontsize=9)
    ax_right.set_ylim(-0.08, 0.08)
    ax_right.tick_params(labelsize=8)

    # Annotate
    ax_left.text(0.5, 0.02, f"Δ = {mean_diff:+.004f}\n95% CI [{ci_lo:.004f}, {ci_hi:.004f}]",
                 transform=ax_left.transAxes, ha="center", fontsize=8, style="italic", color="#555")

    if ax_idx == 0:
        ax_left.set_ylabel("MSCI")

fig.suptitle("RQ2: Planning Mode Differences (Gardner-Altman Estimation)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_rq2_estimation_f.pdf")
fig.savefig(FIG_DIR / "fig3_rq2_estimation_f.png")
plt.close(fig)
print(f"  → Saved: fig3_rq2_estimation_f.pdf/png")


# ═══════════════════════════════════════════════════════════
# FIGURE 4: Forest Plot — All Effect Sizes at a Glance
# ═══════════════════════════════════════════════════════════
print("[Fig 4] Forest plot...")

def d_ci(d_val, n, alpha=0.05):
    se = np.sqrt(1/n + d_val**2 / (2*n))
    z = stats.norm.ppf(1 - alpha/2)
    return d_val - z*se, d_val + z*se

effects = [
    # (label, d, n, group)
    ("RQ1 skip: vs Wrong Image", 2.27, 30, "RQ1"),
    ("RQ1 skip: vs Wrong Audio", 3.64, 30, "RQ1"),
    ("RQ1 full: vs Wrong Image", 2.11, 30, "RQ1"),
    ("RQ1 full: vs Wrong Audio", 2.78, 30, "RQ1"),
    ("RQ1-gen: vs Wrong Image", 4.52, 30, "RQ1-gen"),
    ("RQ1-gen: vs Wrong Audio", 2.02, 30, "RQ1-gen"),
    ("RQ2: Council vs Direct", -0.19, 30, "RQ2"),
    ("RQ2: Ext. Prompt vs Direct", 0.01, 30, "RQ2"),
    ("RQ2: Planner vs Direct", -0.18, 30, "RQ2"),
    ("RQ2-gen: Planner vs Direct", -0.82, 10, "RQ2-gen"),
    ("RQ2-gen: Council vs Direct", -1.40, 10, "RQ2-gen"),
    ("RQ2-gen: Ext. Prompt vs Direct", -1.51, 10, "RQ2-gen"),
]

fig, ax = plt.subplots(figsize=(8, 7.5))

y_positions = list(range(len(effects)))[::-1]
group_colors = {"RQ1": "#2ecc71", "RQ1-gen": "#e67e22", "RQ2": "#3498db", "RQ2-gen": "#e74c3c"}

for i, (label, d, n, group) in enumerate(effects):
    y = y_positions[i]
    lo, hi = d_ci(d, n)
    color = group_colors[group]

    ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", markersize=8,
                color=color, capsize=4, capthick=1.2, linewidth=1.5, zorder=3)
    ax.text(-2.9, y, label, ha="right", va="center", fontsize=9)

# Zero line
ax.axvline(0, color="black", linewidth=0.8, linestyle="-", zorder=1)
# Cohen's thresholds
for threshold, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
    ax.axvline(threshold, color="#bdc3c7", linewidth=0.5, linestyle=":", zorder=0)
    ax.axvline(-threshold, color="#bdc3c7", linewidth=0.5, linestyle=":", zorder=0)

ax.axvline(0.5, color="#bdc3c7", linewidth=0.5, linestyle=":", zorder=0)
ax.text(0.5, len(effects) - 0.3, "medium", ha="center", fontsize=7, color="#999")
ax.text(0.8, len(effects) - 0.3, "large", ha="center", fontsize=7, color="#999")

# Separators between groups
ax.axhline(7.5, color="#bdc3c7", linewidth=0.5, linestyle="-")  # RQ1 vs RQ1-gen
ax.axhline(5.5, color="#bdc3c7", linewidth=0.5, linestyle="-")  # RQ1-gen vs RQ2
ax.axhline(2.5, color="#bdc3c7", linewidth=0.5, linestyle="-")  # RQ2 vs RQ2-gen

ax.set_yticks([])
ax.set_xlabel("Cohen's d [95% CI]")
ax.set_title("Forest Plot: Effect Sizes Across All Comparisons")
ax.set_xlim(-3.0, 6.5)

# Legend
patches = [mpatches.Patch(color=group_colors["RQ1"], label="RQ1 Retrieval"),
           mpatches.Patch(color=group_colors["RQ1-gen"], label="RQ1 Generative (Hybrid)"),
           mpatches.Patch(color=group_colors["RQ2"], label="RQ2 Planning (Retrieval)"),
           mpatches.Patch(color=group_colors["RQ2-gen"], label="RQ2 Planning (Generative)")]
ax.legend(handles=patches, loc="lower right", frameon=True, framealpha=0.9)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_forest_plot_f.pdf")
fig.savefig(FIG_DIR / "fig4_forest_plot_f.png")
plt.close(fig)
print(f"  → Saved: fig4_forest_plot_f.pdf/png")


# ═══════════════════════════════════════════════════════════
# FIGURE 5: Channel Decomposition — Stacked Bars
# ═══════════════════════════════════════════════════════════
print("[Fig 5] Channel decomposition...")

fig, ax = plt.subplots(figsize=(7, 4.5))

conditions = ["Baseline", "Wrong Image", "Wrong Audio"]
cond_keys = ["baseline", "wrong_image", "wrong_audio"]

st_i_means = []
st_a_means = []
for ck in cond_keys:
    sub = rq1_skip_agg[rq1_skip_agg["condition"] == ck]
    st_i_means.append(sub["st_i"].mean())
    st_a_means.append(sub["st_a"].mean())

x = np.arange(len(conditions))
width = 0.5

bars1 = ax.bar(x, st_i_means, width, label="Text–Image (CLIP)", color="#e74c3c", alpha=0.8)
bars2 = ax.bar(x, st_a_means, width, bottom=st_i_means, label="Text–Audio (CLAP)", color="#3498db", alpha=0.8)

# Value labels
for i in range(len(conditions)):
    # st_i label
    ax.text(x[i], st_i_means[i] / 2, f"{st_i_means[i]:.3f}", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    # st_a label
    ax.text(x[i], st_i_means[i] + st_a_means[i] / 2, f"{st_a_means[i]:.3f}", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    # Total MSCI
    total = 0.45 * st_i_means[i] + 0.45 * st_a_means[i]
    ax.text(x[i], st_i_means[i] + st_a_means[i] + 0.02, f"MSCI={total:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel("Cosine Similarity")
ax.set_title("RQ1: Channel Decomposition — Text-Image vs Text-Audio Contributions")
ax.legend(loc="upper right")
ax.set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_rq1_channel_decomposition_f.pdf")
fig.savefig(FIG_DIR / "fig5_rq1_channel_decomposition_f.png")
plt.close(fig)
print(f"  → Saved: fig5_rq1_channel_decomposition_f.pdf/png")


# ═══════════════════════════════════════════════════════════
# FIGURE 6: Domain × Condition Heatmap
# ═══════════════════════════════════════════════════════════
print("[Fig 6] Domain heatmap...")

pivot = rq1_skip_agg.pivot_table(values="msci", index="domain", columns="condition", aggfunc="mean")
# Reorder
col_order = ["baseline", "wrong_image", "wrong_audio"]
pivot = pivot[col_order]
pivot.columns = ["Baseline", "Wrong Image", "Wrong Audio"]

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, linewidths=0.5,
            vmin=0.10, vmax=0.50, cbar_kws={"label": "MSCI"})
ax.set_title("RQ1: Mean MSCI by Domain × Condition")
ax.set_ylabel("Domain")
ax.set_xlabel("Condition")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_rq1_domain_heatmap_f.pdf")
fig.savefig(FIG_DIR / "fig6_rq1_domain_heatmap_f.png")
plt.close(fig)
print(f"  → Saved: fig6_rq1_domain_heatmap_f.pdf/png")


# ═══════════════════════════════════════════════════════════
# FIGURE 7: Power Curve for RQ2
# ═══════════════════════════════════════════════════════════
print("[Fig 7] Power curve...")

n = 30
alpha = 0.05
df = n - 1
t_crit_val = t_dist.ppf(1 - alpha/2, df)

d_range = np.arange(0.01, 1.5, 0.01)
powers = []
for d in d_range:
    ncp = d * np.sqrt(n)
    power = 1 - nct.cdf(t_crit_val, df, ncp) + nct.cdf(-t_crit_val, df, ncp)
    powers.append(power)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(d_range, powers, color="#2c3e50", linewidth=2)
ax.fill_between(d_range, powers, alpha=0.1, color="#2c3e50")

# Reference lines
ax.axhline(0.80, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.7)
ax.text(1.4, 0.81, "80% power", ha="right", va="bottom", fontsize=9, color="#e74c3c")
ax.axhline(0.90, color="#e67e22", linewidth=0.8, linestyle="--", alpha=0.7)
ax.text(1.4, 0.91, "90% power", ha="right", va="bottom", fontsize=9, color="#e67e22")

# Mark observed effect sizes
observed = [
    (0.19, "Council\nd = 0.19"),
    (0.18, "Planner\nd = 0.18"),
    (0.01, "Ext. Prompt\nd = 0.01"),
]
for d_obs, label in observed:
    p_obs = float(np.interp(d_obs, d_range, powers))
    ax.scatter([d_obs], [p_obs], c="#e74c3c", s=50, zorder=5)
    ax.annotate(label, (d_obs, p_obs), textcoords="offset points",
                xytext=(15, 10), fontsize=8, color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.8))

# Mark detectable d at 80% power
d80 = d_range[np.argmin(np.abs(np.array(powers) - 0.80))]
ax.axvline(d80, color="#bdc3c7", linewidth=0.8, linestyle=":")
ax.text(d80, 0.05, f"d = {d80:.2f}", ha="center", fontsize=9, color="#555")

# Cohen's thresholds
for thresh, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
    ax.axvline(thresh, color="#ecf0f1", linewidth=6, alpha=0.5, zorder=0)
    ax.text(thresh, 0.02, lbl, ha="center", fontsize=8, color="#bdc3c7")

ax.set_xlabel("Effect Size (Cohen's d)")
ax.set_ylabel("Statistical Power")
ax.set_title("RQ2: Sensitivity Analysis — Power at N = 30")
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.02)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig7_rq2_power_curve_f.pdf")
fig.savefig(FIG_DIR / "fig7_rq2_power_curve_f.png")
plt.close(fig)
print(f"  → Saved: fig7_rq2_power_curve_f.pdf/png")


# ═══════════════════════════════════════════════════════════
# FIGURE 8: Bootstrap Distribution of Mean Differences (RQ1)
# ═══════════════════════════════════════════════════════════
print("[Fig 8] Bootstrap distributions...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
rng = np.random.default_rng(42)

for ax_idx, (perturb, label, color) in enumerate([
    ("wrong_image", "Wrong Image", "#e74c3c"),
    ("wrong_audio", "Wrong Audio", "#3498db"),
]):
    ax = axes[ax_idx]
    base = rq1_skip_agg[rq1_skip_agg["condition"] == "baseline"].set_index("prompt_id")["msci"]
    pert = rq1_skip_agg[rq1_skip_agg["condition"] == perturb].set_index("prompt_id")["msci"]
    common = base.index.intersection(pert.index)
    diff = (base.loc[common] - pert.loc[common]).values

    # Bootstrap 10000 resamples
    boot_means = []
    for _ in range(10000):
        sample = rng.choice(diff, size=len(diff), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)

    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    obs_mean = diff.mean()

    ax.hist(boot_means, bins=60, color=color, alpha=0.6, edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(obs_mean, color="black", linewidth=1.5, linestyle="-", label=f"Observed Δ = {obs_mean:.4f}")
    ax.axvline(ci_lo, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axvline(ci_hi, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

    # Shade CI
    ax.axvspan(ci_lo, ci_hi, alpha=0.15, color="black", label=f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    ax.set_xlabel("Mean Difference (Baseline − " + label + ")")
    ax.set_ylabel("Density")
    ax.set_title(f"Bootstrap: Baseline vs {label}")
    ax.legend(fontsize=8, loc="upper left" if ax_idx == 1 else "upper right")

fig.suptitle("RQ1: Bootstrap Distributions of Mean MSCI Differences (10,000 resamples)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig8_rq1_bootstrap_f.pdf")
fig.savefig(FIG_DIR / "fig8_rq1_bootstrap_f.png")
plt.close(fig)
print(f"  → Saved: fig8_rq1_bootstrap_f.pdf/png")


# ═══════════════════════════════════════════════════════════
# FIGURE 9: Skip-Text vs Full-Pipeline Comparison (Robustness)
# ═══════════════════════════════════════════════════════════
print("[Fig 9] Skip-text vs full-pipeline robustness comparison...")

fig, ax = plt.subplots(figsize=(8, 4.5))

# Data for grouped bars
modes = ["Skip-Text", "Full Pipeline"]
conds = ["Baseline", "Wrong Image", "Wrong Audio"]
cond_keys_list = ["baseline", "wrong_image", "wrong_audio"]
cond_colors = [PAL_RQ1[c] for c in conds]

means_skip = [rq1_skip_agg[rq1_skip_agg["condition"] == ck]["msci"].mean() for ck in cond_keys_list]
means_full = [rq1_full_agg[rq1_full_agg["condition"] == ck]["msci"].mean() for ck in cond_keys_list]
sems_skip = [rq1_skip_agg[rq1_skip_agg["condition"] == ck]["msci"].sem() for ck in cond_keys_list]
sems_full = [rq1_full_agg[rq1_full_agg["condition"] == ck]["msci"].sem() for ck in cond_keys_list]

x = np.arange(len(conds))
width = 0.35

bars1 = ax.bar(x - width/2, means_skip, width, yerr=sems_skip, capsize=4,
               color=[c for c in cond_colors], alpha=0.9, edgecolor="black", linewidth=0.5, label="Skip-Text")
bars2 = ax.bar(x + width/2, means_full, width, yerr=sems_full, capsize=4,
               color=[c for c in cond_colors], alpha=0.5, edgecolor="black", linewidth=0.5,
               hatch="///", label="Full Pipeline")

# Value labels
for bar_group, means in [(bars1, means_skip), (bars2, means_full)]:
    for bar, val in zip(bar_group, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(conds)
ax.set_ylabel("MSCI (mean ± SE)")
ax.set_title("RQ1 Robustness: Skip-Text vs Full Pipeline")
ax.legend()
ax.set_ylim(0, 0.52)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig9_rq1_robustness_f.pdf")
fig.savefig(FIG_DIR / "fig9_rq1_robustness_f.png")
plt.close(fig)
print(f"  → Saved: fig9_rq1_robustness_f.pdf/png")


# ═══════════════════════════════════════════════════════════
# FIGURE 10: Seed Stability — Within-Prompt Variance
# ═══════════════════════════════════════════════════════════
print("[Fig 10] Seed stability...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# RQ1 full-pipeline (skip-text has zero seed variance by design)
rq1_seed_var = rq1_full.groupby(["prompt_id", "condition"])["msci"].std().reset_index()
rq1_seed_var.columns = ["prompt_id", "condition", "within_sd"]
rq1_seed_var["Condition"] = rq1_seed_var["condition"].map(cond_map)

ax = axes[0]
for cond, color in PAL_RQ1.items():
    data = rq1_seed_var[rq1_seed_var["Condition"] == cond]["within_sd"].dropna()
    ax.hist(data, bins=np.arange(0, 0.16, 0.01), alpha=0.6, color=color, label=cond, edgecolor="white", linewidth=0.3)
ax.set_xlabel("Within-Prompt SD (across 3 seeds)")
ax.set_ylabel("Count")
ax.set_title("RQ1 Full Pipeline")
ax.legend(fontsize=9)
ax.set_xlim(0, 0.15)
# Add median annotation
for cond, color in [("Baseline", PAL_RQ1["Baseline"]), ("Wrong Image", PAL_RQ1["Wrong Image"]), ("Wrong Audio", PAL_RQ1["Wrong Audio"])]:
    med = rq1_seed_var[rq1_seed_var["Condition"] == cond]["within_sd"].median()
    ax.axvline(med, color=color, linewidth=1.2, linestyle="--", alpha=0.7)

# RQ2: within-prompt SD across seeds
rq2_seed_var = rq2.groupby(["prompt_id", "mode"])["msci"].std().reset_index()
rq2_seed_var.columns = ["prompt_id", "mode", "within_sd"]
mode_map = {"direct": "Direct", "council": "Council", "extended_prompt": "Extended Prompt", "planner": "Planner"}
rq2_seed_var["Mode"] = rq2_seed_var["mode"].map(mode_map)

ax = axes[1]
for mode_name, color in PAL_RQ2.items():
    data = rq2_seed_var[rq2_seed_var["Mode"] == mode_name]["within_sd"].dropna()
    ax.hist(data, bins=np.arange(0, 0.16, 0.01), alpha=0.5, color=color, label=mode_name, edgecolor="white", linewidth=0.3)
ax.set_xlabel("Within-Prompt SD (across 3 seeds)")
ax.set_ylabel("Count")
ax.set_title("RQ2 Planning Modes")
ax.legend(fontsize=9)
ax.set_xlim(0, 0.15)

fig.suptitle("Reproducibility: Within-Prompt Variance Across Random Seeds", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig10_seed_stability_f.pdf")
fig.savefig(FIG_DIR / "fig10_seed_stability_f.png")
plt.close(fig)
print(f"  → Saved: fig10_seed_stability_f.pdf/png")


# ═══════════════════════════════════════════════════════════
# FIGURE 11: Retrieval vs Generation Comparison (if data exists)
# ═══════════════════════════════════════════════════════════
# Prefer hybrid results over pure generative
rq1_gen_path = ROOT / "runs/rq1_hybrid/rq1_hybrid_results.json"
if not rq1_gen_path.exists():
    rq1_gen_path = ROOT / "runs/rq1_gen/rq1_gen_results.json"
if rq1_gen_path.exists():
    print("[Fig 13] Retrieval vs Generation comparison...")

    rq1_gen = load_rq1(rq1_gen_path)
    rq1_gen_agg = rq1_gen.groupby(["prompt_id", "domain", "condition"]).mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Side-by-side MSCI by condition
    ax = axes[0]
    cond_keys_list = ["baseline", "wrong_image", "wrong_audio"]
    cond_labels = ["Baseline", "Wrong Image", "Wrong Audio"]
    cond_colors = [PAL_RQ1[c] for c in cond_labels]

    means_ret = [rq1_skip_agg[rq1_skip_agg["condition"] == ck]["msci"].mean() for ck in cond_keys_list]
    means_gen = [rq1_gen_agg[rq1_gen_agg["condition"] == ck]["msci"].mean() for ck in cond_keys_list]
    sems_ret = [rq1_skip_agg[rq1_skip_agg["condition"] == ck]["msci"].sem() for ck in cond_keys_list]
    sems_gen = [rq1_gen_agg[rq1_gen_agg["condition"] == ck]["msci"].sem() for ck in cond_keys_list]

    x = np.arange(len(cond_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, means_ret, width, yerr=sems_ret, capsize=4,
                   color=cond_colors, alpha=0.9, edgecolor="black", linewidth=0.5, label="Retrieval")
    bars2 = ax.bar(x + width/2, means_gen, width, yerr=sems_gen, capsize=4,
                   color=cond_colors, alpha=0.5, edgecolor="black", linewidth=0.5,
                   hatch="///", label="Generative")

    for bar_group, means in [(bars1, means_ret), (bars2, means_gen)]:
        for bar, val in zip(bar_group, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("MSCI (mean ± SE)")
    ax.set_title("(a) Mean MSCI by Condition")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(means_ret), max(means_gen)) * 1.35)

    # Panel B: Effect size comparison
    ax = axes[1]

    def _compute_d(df_agg, baseline_cond, perturb_cond):
        base = df_agg[df_agg["condition"] == baseline_cond].set_index("prompt_id")["msci"]
        pert = df_agg[df_agg["condition"] == perturb_cond].set_index("prompt_id")["msci"]
        common = base.index.intersection(pert.index)
        diff = (base.loc[common] - pert.loc[common]).values
        if len(diff) == 0 or np.std(diff) == 0:
            return 0.0, len(diff)
        return float(np.mean(diff) / np.std(diff)), len(diff)

    comparisons = [
        ("vs Wrong Image", "wrong_image"),
        ("vs Wrong Audio", "wrong_audio"),
    ]

    y_pos = np.arange(len(comparisons) * 2)
    labels = []
    d_vals = []
    colors_bar = []

    for ci, (label, perturb) in enumerate(comparisons):
        d_ret, n_ret = _compute_d(rq1_skip_agg, "baseline", perturb)
        d_gen, n_gen = _compute_d(rq1_gen_agg, "baseline", perturb)

        labels.extend([f"Retrieval: {label}", f"Generative: {label}"])
        d_vals.extend([d_ret, d_gen])
        colors_bar.extend(["#2ecc71", "#e67e22"])

    y_pos = np.arange(len(labels))[::-1]

    for i, (d, lbl, col) in enumerate(zip(d_vals, labels, colors_bar)):
        se = np.sqrt(1/30 + d**2 / 60)
        z = 1.96
        ax.barh(y_pos[i], d, color=col, alpha=0.7, edgecolor="black", linewidth=0.5, height=0.6)
        ax.errorbar(d, y_pos[i], xerr=z*se, fmt="none", ecolor="black", capsize=3, linewidth=1.0)
        ax.text(d + z*se + 0.1, y_pos[i], f"d = {d:.2f}", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.8, color="#bdc3c7", linewidth=0.5, linestyle=":", zorder=0)
    ax.text(0.8, max(y_pos) + 0.5, "large", ha="center", fontsize=8, color="#999")
    ax.set_xlabel("Cohen's d [95% CI]")
    ax.set_title("(b) Effect Sizes: Retrieval vs Generative")

    fig.suptitle("RQ1: Retrieval vs Generative Mode Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig13_retrieval_vs_generation_f.pdf")
    fig.savefig(FIG_DIR / "fig13_retrieval_vs_generation_f.png")
    plt.close(fig)
    print(f"  → Saved: fig13_retrieval_vs_generation_f.pdf/png")
else:
    print("\n[Fig 13] Skipped — no generative results found at runs/rq1_gen/rq1_gen_results.json")


print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"Output directory: {FIG_DIR}")
print(f"Files: {len(list(FIG_DIR.glob('fig*.pdf')))} PDFs + {len(list(FIG_DIR.glob('fig*.png')))} PNGs + 1 statistical supplement")
