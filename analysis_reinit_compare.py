"""
R1-Major1: compare the frequency-dependent functional differentiation across three
time-constant initializations:
  - baseline : all neurons initialized to tau = 50 ms (the original 20 networks)
  - loguniform : per-neuron tau ~ log-uniform[5, 300] ms
  - const200 : all neurons initialized to tau = 200 ms
Recomputes, for every network, the learned tau distribution and the per-band
matched-mode amplitude statistics (skewness, Gini) and the tau-amplitude Spearman
correlation, using the numpy forward model (no TF). Non-converged runs (min_loss
above a threshold) are excluded. Produces comparison figures + a summary table.

Output -> reinit_compare/
"""
import os, json, glob
import numpy as np
from scipy import stats, signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis_ablation import build_inputs, sim_full_R, interval_index, N_IN, MODES, BANDS, FS

OUT = 'reinit_compare'
os.makedirs(OUT, exist_ok=True)
WK = {'alpha': 'rnn/alpha:0',
      'kernel': 'rnn/atclrnn_model/rnn/simple_rnn_cell/kernel:0',
      'wrec': 'rnn/atclrnn_model/rnn/simple_rnn_cell/recurrent_kernel:0',
      'bias': 'rnn/atclrnn_model/rnn/simple_rnn_cell/bias:0'}
COLORS = {'theta': '#E67E22', 'alpha': '#C0392B', 'beta': '#16A085', 'gamma': '#8E44AD'}
COND_COLOR = {'baseline (50 ms)': '#444444', 'loguniform (5-300 ms)': '#1b9e77', 'const (200 ms)': '#7570b3'}
import h5py


def load_weights(run_path):
    ck = os.path.join(run_path, 'results', 'checkpoints', 'best_model.weights.h5')
    with h5py.File(ck, 'r') as f:
        alpha = np.array(f[WK['alpha']][()]).flatten()
        kernel = np.array(f[WK['kernel']][()])
        W_rec = np.array(f[WK['wrec']][()])
        bias = np.array(f[WK['bias']][()]).flatten()
    return alpha.astype(np.float64), kernel[:N_IN].astype(np.float64), W_rec.astype(np.float64), bias.astype(np.float64)


def min_loss(run_path):
    # all runs (baseline and reinit) store the loss history here
    hj = os.path.join(run_path, 'results', 'history.json')
    if os.path.exists(hj):
        h = json.load(open(hj))
        if 'loss' in h and len(h['loss']):
            return float(np.min(h['loss']))
    return np.nan


def gini(a):
    a = np.sort(np.abs(np.asarray(a).ravel())); n = len(a); idx = np.arange(1, n + 1)
    return np.sum((2 * idx - n - 1) * a) / (n * np.sum(a) + 1e-12)


def bandpass(x, lo, hi):
    b, a = sig.butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    return sig.filtfilt(b, a, x, axis=-1)


def analyze(run_path, period=6000):
    alpha, W_in, W_rec, bias = load_weights(run_path)
    tau = 1.0 / alpha
    u, T = build_inputs(period)
    R = sim_full_R(u, alpha, W_in, W_rec, bias)   # (T, N)
    iv = interval_index(T, period)
    out = {'tau': tau, 'skew': {}, 'gini': {}, 'rho_amp': {}}
    for mi, m in enumerate(MODES):
        seg = R[np.where(iv == mi)[0]][500:].T          # (N, time) matched interval
        lo, hi = BANDS[m]
        filt = bandpass(seg, lo, hi)
        amp = np.abs(sig.hilbert(filt, axis=-1)).mean(-1)   # matched-mode amplitude (N,)
        out['skew'][m] = float(stats.skew(amp))
        out['gini'][m] = float(gini(amp))
        out['rho_amp'][m] = float(stats.spearmanr(tau, amp)[0])
    return out


CONDITIONS = {
    'baseline (50 ms)': [(f'multiple_runs/{i}', i) for i in range(1, 21)],
    'loguniform (5-300 ms)': [(f'multiple_runs_reinit/loguniform/{i}', i) for i in range(1, 11)],
    'const (200 ms)': [(f'multiple_runs_reinit/const200/{i}', i) for i in range(1, 11)],
}
LOSS_THRESH = 2e-5   # exclude non-converged runs

results = {}
for cond, runs in CONDITIONS.items():
    per = []
    excluded = []
    for path, rid in runs:
        if not os.path.exists(os.path.join(path, 'results', 'checkpoints', 'best_model.weights.h5')):
            continue
        ml = min_loss(path)
        if not (ml < LOSS_THRESH):
            excluded.append((rid, ml)); continue
        per.append(analyze(path))
    results[cond] = per
    print(f'{cond}: {len(per)} converged runs used' + (f'  (excluded: {excluded})' if excluded else ''))

# ---------- aggregate ----------
def agg(cond, key, mode):
    return np.array([r[key][mode] for r in results[cond]])

# Figure 1: tau-amp correlation vs band, per condition
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
metrics = [('rho_amp', r'$\rho_s^{\mathrm{amp}}(\tau, \bar{A})$', (-1, 0.1)),
           ('skew', 'amplitude skewness', None),
           ('gini', 'amplitude Gini', (0, 0.7))]
x = np.arange(4)
for ax, (key, ylab, ylim) in zip(axes, metrics):
    for ci, cond in enumerate(CONDITIONS):
        means = [agg(cond, key, m).mean() for m in MODES]
        sds = [agg(cond, key, m).std() for m in MODES]
        ax.errorbar(x + (ci - 1) * 0.12, means, yerr=sds, marker='o', ms=5, capsize=3,
                    lw=1.8, color=COND_COLOR[cond], label=cond)
    ax.set_xticks(x); ax.set_xticklabels([m[0].upper() + r'$\%s$' % m for m in MODES] if False else
                                         [r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$'])
    ax.set_ylabel(ylab); ax.set_xlabel('rhythm mode'); ax.grid(alpha=0.25)
    if ylim: ax.set_ylim(*ylim)
axes[0].legend(fontsize=9, loc='lower left')
fig.suptitle('Frequency-dependent functional differentiation is robust to time-constant initialization '
             f'($n$ = {[len(results[c]) for c in CONDITIONS]} converged runs)', y=1.02)
fig.tight_layout()
for e in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'fig_reinit_metrics.{e}'), dpi=200, bbox_inches='tight')
plt.close(fig)

# Figure 2: learned tau distribution per condition (pooled)
fig, ax = plt.subplots(figsize=(8, 5))
for cond in CONDITIONS:
    pooled = np.concatenate([r['tau'] for r in results[cond]])
    ax.hist(np.log10(pooled), bins=40, histtype='step', lw=2, density=True,
            color=COND_COLOR[cond], label=f'{cond} (median {np.median(pooled):.1f} ms)')
ax.set_xlabel(r'learned $\log_{10}\tau$ (ms)'); ax.set_ylabel('density')
ax.axvline(np.log10(50), color='gray', ls=':', lw=1); ax.text(np.log10(50), ax.get_ylim()[1]*0.9, '50 ms', fontsize=8)
ax.axvline(np.log10(200), color='gray', ls=':', lw=1); ax.text(np.log10(200), ax.get_ylim()[1]*0.9, '200 ms', fontsize=8)
ax.legend(fontsize=9); ax.set_title('Learned time-constant distribution by initialization (pooled over runs)')
fig.tight_layout()
for e in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'fig_reinit_tau.{e}'), dpi=200, bbox_inches='tight')
plt.close(fig)

# ---------- summary table ----------
summary = {}
for cond in CONDITIONS:
    pooled = np.concatenate([r['tau'] for r in results[cond]])
    summary[cond] = {'n_runs': len(results[cond]),
                     'median_tau': float(np.median(pooled)),
                     'frac_tau_lt20': float((pooled < 20).mean()),
                     'frac_tau_gt100': float((pooled > 100).mean())}
    for m in MODES:
        summary[cond][f'rho_amp_{m}'] = [float(agg(cond, 'rho_amp', m).mean()), float(agg(cond, 'rho_amp', m).std())]
        summary[cond][f'gini_{m}'] = [float(agg(cond, 'gini', m).mean()), float(agg(cond, 'gini', m).std())]
        summary[cond][f'skew_{m}'] = [float(agg(cond, 'skew', m).mean()), float(agg(cond, 'skew', m).std())]
json.dump(summary, open(os.path.join(OUT, 'reinit_compare_summary.json'), 'w'), indent=2)

print('\n=== rho_amp (tau, matched amplitude) mean per band ===')
print(f"{'condition':<24} " + '  '.join(f'{m:>7}' for m in MODES))
for cond in CONDITIONS:
    print(f'{cond:<24} ' + '  '.join(f'{agg(cond,"rho_amp",m).mean():+.2f}' for m in MODES))
print('\n=== Gini per band ===')
for cond in CONDITIONS:
    print(f'{cond:<24} ' + '  '.join(f'{agg(cond,"gini",m).mean():.2f}' for m in MODES))
print(f'\nSaved figures + summary to {OUT}/')
