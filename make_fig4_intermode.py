#!/usr/bin/env python3
"""Generate Figure 4: Inter-mode population similarity.

(A) Mean Spearman correlation matrix of dominant-mode amplitudes.
(B) Mean Jaccard coefficient matrix (top-20% sync degree).
(C) Strip + box plot of trial-to-trial variability for each mode pair.
"""

import json
import pathlib
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---------- config ----------
N_RUNS = 20
RUN_DIR = pathlib.Path("multiple_runs")
OUT_PATH = pathlib.Path("figures/fig4_intermode.pdf")
BANDS = ["theta", "alpha", "beta", "gamma"]
BAND_LABELS = [r"$\theta$", r"$\alpha$", r"$\beta$", r"$\gamma$"]
PAIR_LABELS_SHORT = [
    r"$\theta$-$\alpha$",
    r"$\theta$-$\beta$",
    r"$\theta$-$\gamma$",
    r"$\alpha$-$\beta$",
    r"$\alpha$-$\gamma$",
    r"$\beta$-$\gamma$",
]

# ---------- load data ----------
pairs = []
for i in range(4):
    for j in range(i + 1, 4):
        pairs.append((i, j))

corr_all = np.zeros((N_RUNS, 4, 4))
jacc_all = np.zeros((N_RUNS, 4, 4))

for rid in range(N_RUNS):
    p = RUN_DIR / str(rid + 1) / "figures" / "single" / "analysis_results.json"
    with open(p) as f:
        ar = json.load(f)
    # Spearman inter-mode correlation of matched-mode amplitude vectors,
    # recomputed from the stored per-neuron amplitudes (the previously stored
    # dominant_amp_correlation_matrix used Pearson; the manuscript defines and
    # labels this quantity as Spearman -- Eq. amp_corr).
    matched = np.array([np.array(ar["filtered_amps"][BANDS[i]])[i] for i in range(4)])
    corr_all[rid] = spearmanr(matched, axis=1)[0]
    jacc_all[rid] = np.array(ar["jaccard_matrix"])

corr_mean = corr_all.mean(axis=0)
jacc_mean = jacc_all.mean(axis=0)

# Collect pair-wise values for panel C
corr_pairs = {}  # {pair_idx: [values]}
jacc_pairs = {}
for k, (i, j) in enumerate(pairs):
    corr_pairs[k] = corr_all[:, i, j]
    jacc_pairs[k] = jacc_all[:, i, j]

# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.6]})
plt.subplots_adjust(wspace=0.55)

# --- Panel A: Correlation matrix ---
ax = axes[0]
norm_a = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=1.0)
im_a = ax.imshow(corr_mean, cmap="RdBu_r", norm=norm_a, aspect="equal")
ax.set_xticks(range(4))
ax.set_xticklabels(BAND_LABELS, fontsize=12)
ax.set_yticks(range(4))
ax.set_yticklabels(BAND_LABELS, fontsize=12)
# Annotate values
for i in range(4):
    for j in range(4):
        v = corr_mean[i, j]
        color = "white" if abs(v) > 0.6 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color=color)
cb_a = fig.colorbar(im_a, ax=ax, fraction=0.046, pad=0.04)
cb_a.set_label("Spearman $\\rho$", fontsize=10)
ax.set_title("A  Amplitude correlation", fontsize=12, loc="left", fontweight="bold")

# --- Panel B: Jaccard matrix ---
ax = axes[1]
im_b = ax.imshow(jacc_mean, cmap="YlOrRd", vmin=0, vmax=0.7, aspect="equal")
ax.set_xticks(range(4))
ax.set_xticklabels(BAND_LABELS, fontsize=12)
ax.set_yticks(range(4))
ax.set_yticklabels(BAND_LABELS, fontsize=12)
for i in range(4):
    for j in range(4):
        v = jacc_mean[i, j]
        color = "white" if v > 0.45 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color=color)
cb_b = fig.colorbar(im_b, ax=ax, fraction=0.046, pad=0.04)
cb_b.set_label("Jaccard $J$", fontsize=10)
ax.set_title("B  Jaccard coefficient", fontsize=12, loc="left", fontweight="bold")

# --- Panel C: Strip + Box plot ---
ax = axes[2]
n_pairs = len(pairs)
x_corr = np.arange(n_pairs) * 2.5
x_jacc = x_corr + 0.8
width = 0.55

# Box plots — correlation
bp_corr = ax.boxplot(
    [corr_pairs[k] for k in range(n_pairs)],
    positions=x_corr,
    widths=width,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.5),
    boxprops=dict(facecolor="#6baed6", alpha=0.5),
    whiskerprops=dict(color="#2171b5"),
    capprops=dict(color="#2171b5"),
)

# Box plots — Jaccard
bp_jacc = ax.boxplot(
    [jacc_pairs[k] for k in range(n_pairs)],
    positions=x_jacc,
    widths=width,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.5),
    boxprops=dict(facecolor="#fc8d59", alpha=0.5),
    whiskerprops=dict(color="#e6550d"),
    capprops=dict(color="#e6550d"),
)

# Strip (jitter) overlay
rng = np.random.default_rng(42)
for k in range(n_pairs):
    jitter_c = rng.uniform(-0.12, 0.12, size=N_RUNS)
    jitter_j = rng.uniform(-0.12, 0.12, size=N_RUNS)
    ax.scatter(x_corr[k] + jitter_c, corr_pairs[k], s=22, color="#2171b5",
               alpha=0.8, zorder=5, edgecolors="white", linewidths=0.3)
    ax.scatter(x_jacc[k] + jitter_j, jacc_pairs[k], s=22, color="#e6550d",
               alpha=0.8, zorder=5, edgecolors="white", linewidths=0.3)

ax.set_xticks((x_corr + x_jacc) / 2)
ax.set_xticklabels(PAIR_LABELS_SHORT, fontsize=11)
ax.set_ylabel("Similarity", fontsize=11)
ax.set_ylim(-0.3, 1.05)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#6baed6", alpha=0.5, edgecolor="gray", label="Amp. corr. $\\rho$"),
    Patch(facecolor="#fc8d59", alpha=0.5, edgecolor="gray", label="Jaccard $J$"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper left")
ax.set_title("C  Run-to-run variability", fontsize=12, loc="left", fontweight="bold")

fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
fig.savefig(OUT_PATH.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"Saved to {OUT_PATH} and {OUT_PATH.with_suffix('.png')}")
plt.close()
