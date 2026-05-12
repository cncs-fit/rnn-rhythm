"""
Supplementary Figure: κ̄ distribution and κ-η relationship across 20 runs
- (A) κ̄ vs η^(β,α) scatter (classification boundary)
- (B) Histogram of mean κ per run
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
})

# Load results
with open("multiple_runs/summary/phase_coherence_results.json") as f:
    data = json.load(f)

run_ids = []
kappa_means = []
eta_beta_alpha = []
for entry in data["per_run"]:
    run_ids.append(entry["run_id"])
    kappa_means.append(entry["kappa_mean"])
    eta_beta_alpha.append(entry["eta_beta_alpha"])

run_ids = np.array(run_ids)
kappa_means = np.array(kappa_means)
eta_beta_alpha = np.array(eta_beta_alpha)

threshold = 0.1
phase_mask = kappa_means > threshold
n_phase = np.sum(phase_mask)
n_amp = np.sum(~phase_mask)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- (A) κ̄ vs η^(β,α) scatter ---
ax = axes[0]
ax.scatter(kappa_means[phase_mask], eta_beta_alpha[phase_mask],
           c='steelblue', s=60, edgecolors='k', linewidths=0.5, zorder=3,
           label=f'Phase interference ({n_phase})')
ax.scatter(kappa_means[~phase_mask], eta_beta_alpha[~phase_mask],
           c='tomato', s=60, edgecolors='k', linewidths=0.5, zorder=3,
           label=f'Amplitude suppression ({n_amp})')
# annotate run IDs
for i, rid in enumerate(run_ids):
    ax.annotate(str(rid), (kappa_means[i], eta_beta_alpha[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)
ax.axvline(threshold, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
ax.set_xlabel(r'Mean $\bar{\kappa}$')
ax.set_ylabel(r'$\eta^{(\beta,\alpha)}$')
ax.set_title(r'(A) $\bar{\kappa}$ vs $\eta^{(\beta,\alpha)}$')
ax.legend(fontsize=9, loc='upper right')

# --- (B) Histogram ---
ax = axes[1]
# Log-spaced bins to handle wide range (0.002 -- 5.2)
bins = np.logspace(np.log10(0.001), np.log10(10), 20)
ax.hist(kappa_means[~phase_mask], bins=bins, color='tomato', edgecolor='white',
        alpha=0.85, label=f'Amplitude suppression ({n_amp})')
ax.hist(kappa_means[phase_mask], bins=bins, color='steelblue', edgecolor='white',
        alpha=0.85, label=f'Phase interference ({n_phase})')
ax.axvline(threshold, color='gray', linestyle='--', linewidth=1.0, alpha=0.7,
           label=f'threshold = {threshold}')
ax.set_xscale('log')
ax.set_xlabel(r'Mean $\bar{\kappa}$')
ax.set_ylabel('Count')
ax.set_title(r'(B) Distribution of $\bar{\kappa}$ across 20 runs')
ax.legend(fontsize=9)

plt.tight_layout()

out_dir = "multiple_runs/summary"
plt.savefig(f"{out_dir}/supfig_kappa_distribution.png", dpi=200, bbox_inches='tight')
plt.savefig(f"{out_dir}/supfig_kappa_distribution.pdf", bbox_inches='tight')
print(f"Saved to {out_dir}/supfig_kappa_distribution.pdf")

# (paper-internal: also copy to the manuscript figs directory if it exists)
# import shutil
# shutil.copy(f"{out_dir}/supfig_kappa_distribution.pdf", "paper/figs/supfig_kappa.pdf")
# print("Copied to paper/figs/supfig_kappa.pdf")
