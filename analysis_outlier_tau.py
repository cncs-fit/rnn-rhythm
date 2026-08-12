"""
Inter-mode population sharing vs. the learned time-constant distribution
(manuscript Sect. 3.4 and Appendix C, Fig. 11).

Self-contained version: everything is recomputed from the trained model weights
(read from the .h5 checkpoints) via the numpy forward model in analysis_ablation.py
(no TensorFlow required). For each network it computes the Spearman inter-mode
amplitude-pattern correlations (Eq. amp_corr), the Jaccard overlap of the active
(top-20% sync-degree) populations, and the median learned time constant, then
relates inter-mode sharing to the median time constant and annotates the outlier
networks.

Output -> outlier_analysis/
"""
import os
import json
import numpy as np
from scipy import stats, signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis_ablation import (extract_weights, build_inputs, sim_full_R,
                               interval_index, MODES, BANDS, FS)

RUNS = range(1, 21)
OUT = 'outlier_analysis'
os.makedirs(OUT, exist_ok=True)
COLORS = {'theta': '#E67E22', 'alpha': '#C0392B', 'beta': '#16A085', 'gamma': '#8E44AD'}

# outlier networks (annotated), based on the Spearman inter-mode correlations
HIGH_TB = {12, 16}       # highest theta-beta correlation
LOW_BG = {9, 14, 18}     # lowest beta-gamma correlation


def per_run(run, period=8000, transient=500, top=0.2):
    """Return (matched-mode Spearman corr matrix, Jaccard matrix, tau) for one run."""
    a, W_in, W_rec, b = extract_weights(f'multiple_runs/{run}')
    tau = 1.0 / a
    u, T = build_inputs(period)
    R = sim_full_R(u, a, W_in, W_rec, b)
    iv = interval_index(T, period)
    matched = []          # matched-mode amplitude vector per mode
    active = {}           # top-20% sync-degree set per mode
    for i, m in enumerate(MODES):
        lo, hi = BANDS[m]
        bb, aa = sig.butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='band')
        seg = R[np.where(iv == i)[0]].T          # (N, period), full interval
        filt = sig.filtfilt(bb, aa, seg, axis=-1)
        env = np.abs(sig.hilbert(filt, axis=-1))
        matched.append(env[:, transient:-1].mean(-1))   # matched-mode amplitude
        # absolute sync strength S_ij over the analysis window -> sync degree -> active set
        segw = R[np.where(iv == i)[0]][transient:].T
        fw = sig.filtfilt(bb, aa, segw, axis=-1)
        an = sig.hilbert(fw, axis=-1); e = np.abs(an); p = np.angle(an)
        w = e[:, None, :] * e[None, :, :]; dp = p[:, None, :] - p[None, :, :]
        S = np.abs(np.mean(w * np.exp(1j * dp), axis=-1))
        d = S.sum(1); k = int(round(top * len(d)))
        active[m] = set(np.argsort(-d)[:k])
    C = stats.spearmanr(np.array(matched), axis=1)[0]
    J = np.zeros((4, 4))
    for i, mi in enumerate(MODES):
        for j, mj in enumerate(MODES):
            J[i, j] = len(active[mi] & active[mj]) / len(active[mi] | active[mj])
    return C, J, tau


bg_corr, tb_corr, bg_jac, tb_jac, med_tau, ids = [], [], [], [], [], []
for r in RUNS:
    if not os.path.exists(f'multiple_runs/{r}/results/checkpoints/best_model.weights.h5'):
        continue
    C, J, tau = per_run(r)
    bg_corr.append(C[2, 3]); tb_corr.append(C[0, 2])
    bg_jac.append(J[2, 3]); tb_jac.append(J[0, 2])
    med_tau.append(np.median(tau)); ids.append(r)

bg_corr, tb_corr = np.array(bg_corr), np.array(tb_corr)
bg_jac, tb_jac = np.array(bg_jac), np.array(tb_jac)
med_tau = np.array(med_tau); ids = np.array(ids)


def panel(ax, y, jac, title, annotate):
    rho, pval = stats.spearmanr(med_tau, y)
    sc = ax.scatter(med_tau, y, c=jac, cmap='viridis', s=70,
                    edgecolors='k', linewidths=0.5, zorder=3)
    for xi, yi, rid in zip(med_tau, y, ids):
        if rid in annotate:
            ax.annotate(f'#{rid}', (xi, yi), textcoords='offset points',
                        xytext=(6, 4), fontsize=10, fontweight='bold', color='crimson')
    bfit, afit = np.polyfit(med_tau, y, 1)
    xs = np.linspace(med_tau.min(), med_tau.max(), 50)
    ax.plot(xs, bfit * xs + afit, 'k--', alpha=0.5, lw=1.2, zorder=2)
    ax.set_xlabel('median learned time constant (ms)')
    ax.set_title(f'{title}\nSpearman $\\rho$ = {rho:+.2f}  ($p$ = {pval:.3f})', fontsize=12)
    ax.grid(alpha=0.25)
    return sc


plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc = panel(axes[0], bg_corr, bg_jac, r'$\beta$--$\gamma$ amplitude correlation', LOW_BG)
axes[0].set_ylabel('amplitude-pattern correlation')
cb = fig.colorbar(sc, ax=axes[0]); cb.set_label(r'$\beta$--$\gamma$ Jaccard')
sc2 = panel(axes[1], tb_corr, tb_jac, r'$\theta$--$\beta$ amplitude correlation', HIGH_TB)
cb2 = fig.colorbar(sc2, ax=axes[1]); cb2.set_label(r'$\theta$--$\beta$ Jaccard')
fig.suptitle('Inter-mode population sharing vs. learned median time constant '
             f'(n = {len(ids)} networks)', y=1.02, fontsize=13)
fig.tight_layout()
for e in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'fig_outlier_tau.{e}'), dpi=200, bbox_inches='tight')
plt.close(fig)

summary = {
    'beta_gamma_corr_vs_median_tau': list(stats.spearmanr(med_tau, bg_corr)),
    'theta_beta_corr_vs_median_tau': list(stats.spearmanr(med_tau, tb_corr)),
    'high_theta_beta_outliers': sorted(HIGH_TB),
    'low_beta_gamma_outliers': sorted(LOW_BG),
}
with open(os.path.join(OUT, 'outlier_tau_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print('beta-gamma corr vs median tau: rho=%.2f p=%.3f' % tuple(stats.spearmanr(med_tau, bg_corr)))
print('theta-beta corr vs median tau: rho=%.2f p=%.3f' % tuple(stats.spearmanr(med_tau, tb_corr)))
print('outliers: high theta-beta =', sorted(HIGH_TB), ' low beta-gamma =', sorted(LOW_BG))
print(f'Saved {OUT}/fig_outlier_tau.pdf')
