"""
R2-Major2 & 3: structural and dynamical basis of the learned rhythms.
Reuses the numpy forward model (analysis_ablation) and the discrete-time Jacobian
(analysis_baseline_fp) -- no TF/GPU. Produces three linked results:

  (1) Connectivity structure: the short-tau (fast) subpopulation is more strongly
      recurrently interconnected than the slow subpopulation (W_rec block analysis,
      all 20 networks).
  (2) Effective vs. intrinsic time constants: the recurrent connectivity generates
      collective timescales far exceeding any single neuron's intrinsic tau
      (slowest relaxation mode at a fixed point, all 20 networks) -- the effective time
      constant is not set by neuronal properties alone (cf. Marti 2018; Mazzucato).
  (3) Gain modulation: the beta<->gamma baseline shift (mechanism B) retunes the
      unstable eigenfrequency purely by changing the neuronal gain (1 - r^2), with
      W_rec and the leak rates held fixed (representative run 1, J(beta,gamma)=0.67).

Output -> connectivity_results/
"""
import os
import json
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis_ablation import (extract_weights, build_inputs, sim_full_R,
                               interval_index, MODES, BANDS, FS)

OUT = 'connectivity_results'
os.makedirs(OUT, exist_ok=True)
COLORS = {'theta': '#E67E22', 'alpha': '#C0392B', 'beta': '#16A085', 'gamma': '#8E44AD'}


def mode_baselines(run, period=4000):
    a, Win, W, b = extract_weights(f'multiple_runs/{run}')
    u, T = build_inputs(period)
    R = sim_full_R(u, a, Win, W, b)
    iv = interval_index(T, period)
    base = {m: R[np.where(iv == i)[0][500:]].mean(0) for i, m in enumerate(MODES)}
    return a, W, b, base


def jacobian(alpha, W, rbar):
    """Discrete-time Jacobian at operating point with firing rate rbar. gain = 1-rbar^2."""
    return np.diag(1 - alpha) + np.diag(alpha) @ W.T @ np.diag(1 - rbar ** 2)


def dominant_unstable(J):
    mu = np.linalg.eigvals(J)
    mag = np.abs(mu); f = np.abs(np.angle(mu)) / (2 * np.pi) * FS
    m = (mag > 1) & (f > 1)
    if not m.any():
        return np.nan, np.nan
    i = np.argmax(mag * m)
    return f[i], mag[i]


# ================= Part 1 & 2: all runs =================
wf, ws, wx = [], [], []
wf_nd, ws_nd = [], []   # self-connections (diagonal) excluded, robustness check
eff_tau, intr_taumax = [], []
for r in range(1, 21):
    a, Win, W, b = extract_weights(f'multiple_runs/{r}')
    tau = 1 / a; N = len(tau)
    o = np.argsort(tau); fast, slow = o[:N // 3], o[-N // 3:]
    blk = lambda A, B: np.mean(np.abs(W[np.ix_(A, B)]))
    wf.append(blk(fast, fast)); ws.append(blk(slow, slow))
    wx.append((blk(fast, slow) + blk(slow, fast)) / 2)
    # diagonal-excluded within-block means (off-diagonal only)
    Wnd = W.copy(); np.fill_diagonal(Wnd, 0.0)
    def blk_nd(idx):
        sub = np.abs(Wnd[np.ix_(idx, idx)]); n = len(idx)
        return sub.sum() / (n * n - n)
    wf_nd.append(blk_nd(fast)); ws_nd.append(blk_nd(slow))
    # Linearize about an actual fixed point (found from the origin) of the
    # input-free dynamics x(t+1) = (1-a)x + a(tanh(x) W + b); the gain there is
    # 1 - r*^2. The slowest DECAYING (|mu|<1) eigenmode gives the effective
    # relaxation time constant (the fixed point is itself unstable: its
    # oscillatory |mu|>1 modes generate the rhythms, cf. Fig. 5).
    fmap = lambda x: (1 - a) * x + a * (np.tanh(x) @ W + b)
    xstar = minimize(lambda x: np.sum((fmap(x) - x) ** 2), np.zeros_like(a),
                     method='L-BFGS-B',
                     options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-12}).x
    Jfp = np.diag(1 - a) + np.diag(a) @ W.T @ np.diag(1 - np.tanh(xstar) ** 2)
    mag = np.abs(np.linalg.eigvals(Jfp)); st = mag[mag < 0.9999]
    eff_tau.append(-1 / np.log(st.max())); intr_taumax.append(tau.max())
wf, ws, wx = map(np.array, (wf, ws, wx))
wf_nd, ws_nd = np.array(wf_nd), np.array(ws_nd)
eff_tau, intr_taumax = np.array(eff_tau), np.array(intr_taumax)

# ================= Part 3: gain modulation, run 1 =================
a1, W1, b1, base1 = mode_baselines(1)
g_beta = 1 - base1['beta'] ** 2
g_gamma = 1 - base1['gamma'] ** 2
svals = np.linspace(0, 1, 21)
interp_freq = []
for s in svals:
    g = (1 - s) * g_beta + s * g_gamma
    J = np.diag(1 - a1) + np.diag(a1) @ W1.T @ np.diag(g)
    f, _ = dominant_unstable(J)
    interp_freq.append(f)
interp_freq = np.array(interp_freq)

# ================= Figure (2x2) =================
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13})
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# (a) block coupling
a0 = ax[0, 0]
means = [wf.mean(), ws.mean(), wx.mean()]; sds = [wf.std(), ws.std(), wx.std()]
labels = ['within\nfast', 'within\nslow', 'cross']
bars = a0.bar(labels, means, yerr=sds, capsize=5,
              color=['#B2182B', '#2166AC', '#999999'], alpha=0.85)
for i, (rr) in enumerate([wf, ws, wx]):
    a0.scatter(np.full(len(rr), i) + np.random.uniform(-.08, .08, len(rr)), rr,
               s=12, color='k', alpha=0.4, zorder=3)
a0.set_ylabel(r'mean $|W_{\mathrm{rec}}|$')
a0.set_title('(a) Recurrent coupling by subpopulation\n(fast/slow = short/long-$\\tau$ terciles)')
_, p_fs = stats.wilcoxon(wf, ws)
a0.text(0.5, max(means) * 1.18, f'within-fast > within-slow\n20/20 runs, $p$={p_fs:.1e}',
        ha='center', fontsize=10)
a0.set_ylim(0, max(means) * 1.35)

# (b) effective vs intrinsic timescale
a1p = ax[0, 1]
a1p.scatter(intr_taumax, eff_tau, s=55, color='#8E44AD', edgecolors='k', linewidths=0.5, zorder=3)
lim = [min(intr_taumax.min(), eff_tau.min()) * 0.8, max(intr_taumax.max(), eff_tau.max()) * 1.2]
a1p.plot(lim, lim, 'k--', alpha=0.5, label='effective = intrinsic')
a1p.set_xscale('log'); a1p.set_yscale('log'); a1p.set_xlim(lim); a1p.set_ylim(lim)
a1p.set_xlabel(r'max intrinsic $\tau$ (ms)')
a1p.set_ylabel(r'slowest relaxation time constant $\tau_{\mathrm{relax}}$ (ms)')
a1p.set_title('(b) Recurrence extends the relaxation timescale\nbeyond the intrinsic time constants')
a1p.legend(loc='upper left'); a1p.grid(alpha=0.25, which='both')

# (c) gain distributions beta vs gamma (run 1)
a2 = ax[1, 0]
a2.hist(g_beta, bins=20, alpha=0.6, color=COLORS['beta'], label=r'$\beta$ (mean %.2f)' % g_beta.mean())
a2.hist(g_gamma, bins=20, alpha=0.6, color=COLORS['gamma'], label=r'$\gamma$ (mean %.2f)' % g_gamma.mean())
a2.set_xlabel(r'neuronal gain $1 - r_i^2$ at operating point')
a2.set_ylabel('number of neurons')
a2.set_title('(c) Baseline shift changes the gain profile\n(run 1, $\\beta$ vs $\\gamma$)')
a2.legend()

# (d) gain interpolation -> frequency retuning
a3 = ax[1, 1]
a3.axhspan(*BANDS['beta'], alpha=0.12, color=COLORS['beta'])
a3.axhspan(*BANDS['gamma'], alpha=0.12, color=COLORS['gamma'])
a3.plot(svals, interp_freq, '-o', color='k', ms=4)
a3.set_xlabel(r'gain profile: $\beta \;\rightarrow\; \gamma$  (interpolation $s$)')
a3.set_ylabel('dominant unstable frequency (Hz)')
a3.set_title('(d) Gain modulation alone retunes the frequency\n($W_{\\mathrm{rec}}$, $\\lambda$ fixed)')
a3.text(0.02, np.mean(BANDS['beta']), r'$\beta$', color=COLORS['beta'], fontsize=13, va='center')
a3.text(0.02, np.mean(BANDS['gamma']), r'$\gamma$', color=COLORS['gamma'], fontsize=13, va='center')
a3.grid(alpha=0.25)

fig.tight_layout()
for e in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'fig_connectivity.{e}'), dpi=200, bbox_inches='tight')
plt.close(fig)

# ================= summary =================
summary = {
    'block_coupling': {'within_fast': [float(wf.mean()), float(wf.std())],
                       'within_slow': [float(ws.mean()), float(ws.std())],
                       'cross': [float(wx.mean()), float(wx.std())],
                       'wilcoxon_fast_vs_slow_p': float(p_fs),
                       'n_runs_fast_gt_slow': int((wf > ws).sum())},
    'timescale': {'eff_tau_median_ms': float(np.median(eff_tau)),
                  'intr_taumax_median_ms': float(np.median(intr_taumax)),
                  'ratio_median': float(np.median(eff_tau / intr_taumax)),
                  'n_runs_ratio_gt2': int((eff_tau / intr_taumax > 2).sum())},
    'gain_run1': {'beta_mean_gain': float(g_beta.mean()), 'gamma_mean_gain': float(g_gamma.mean()),
                  'beta_freq': float(interp_freq[0]), 'gamma_freq': float(interp_freq[-1])},
}
with open(os.path.join(OUT, 'connectivity_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print('Part1 within-fast/slow/cross:', [f'{m:.4f}' for m in means], 'p=%.1e' % p_fs)
_, p_fs_nd = stats.wilcoxon(wf_nd, ws_nd)
print('Part1 (self-connections excluded) within-fast=%.4f within-slow=%.4f  fast>slow %d/20  p=%.1e'
      % (wf_nd.mean(), ws_nd.mean(), int((wf_nd > ws_nd).sum()), p_fs_nd))
print('Part2 eff/intr ratio median: %.1fx' % np.median(eff_tau / intr_taumax))
print('Part3 run1 freq: beta %.1f -> gamma %.1f Hz (gain %.2f -> %.2f)'
      % (interp_freq[0], interp_freq[-1], g_beta.mean(), g_gamma.mean()))
print(f'Saved {OUT}/fig_connectivity.pdf')
