"""
analysis_ablation.py
====================
R1-Major2 (causal test): silence subpopulations of neurons in the trained
networks and measure the effect on each rhythm's band power in the population
output. Tests whether high-frequency (beta/gamma) generation causally depends
on short-time-constant (fast) neurons, as the correlational analyses suggest.

Two silencing methods:
  - zero   : ablated neurons' output r_i is set to 0 (removed from recurrent
             dynamics AND readout) -- full necessity test.
  - freeze : ablated neurons' output r_i is clamped to their per-mode baseline
             mean firing rate -- removes their OSCILLATION but keeps their DC /
             baseline contribution, controlling for the baseline-shift
             mechanism (Mechanism B).

Three ranking series (swept over ablation fraction):
  - slow : ablate longest-tau (slowest) neurons first
  - fast : ablate shortest-tau (fastest) neurons first
  - random : random subset (mean over several trials; control)

Everything runs in pure numpy (weights read from .h5 via h5py); no TensorFlow /
GPU is used, so it does not contend with concurrent training jobs. The numpy
forward model reproduces the Keras model's band power exactly (validated).

Outputs -> ablation_results/ (isolated; does not touch existing results).
"""
import os
import json
import argparse
import numpy as np
import h5py
from scipy import signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------- config -----------------------------
N = 100
N_IN = 4
WAIT = 100
PULSE = 100
MODES = ['theta', 'alpha', 'beta', 'gamma']
BANDS = {'theta': (4, 7), 'alpha': (8, 13), 'beta': (14, 29), 'gamma': (30, 50)}
FS = 1000.0
COLORS = {'theta': '#E67E22', 'alpha': '#C0392B', 'beta': '#16A085', 'gamma': '#8E44AD'}
SERIES_STYLE = {'slow': ('tab:blue', 'o', 'slow-first (long tau)'),
                'fast': ('tab:red', 's', 'fast-first (short tau)'),
                'random': ('gray', '^', 'random (control)')}

WKEY = {
    'alpha': 'rnn/alpha:0',
    'kernel': 'rnn/atclrnn_model/rnn/simple_rnn_cell/kernel:0',
    'wrec': 'rnn/atclrnn_model/rnn/simple_rnn_cell/recurrent_kernel:0',
    'bias': 'rnn/atclrnn_model/rnn/simple_rnn_cell/bias:0',
}


def extract_weights(run_path):
    ck = os.path.join(run_path, 'results', 'checkpoints', 'best_model.weights.h5')
    if not os.path.exists(ck):
        ck = os.path.join(run_path, 'results', 'checkpoints', 'last_weight.weights.h5')
    with h5py.File(ck, 'r') as f:
        alpha = np.array(f[WKEY['alpha']][()]).flatten()
        kernel = np.array(f[WKEY['kernel']][()])
        W_rec = np.array(f[WKEY['wrec']][()])
        bias = np.array(f[WKEY['bias']][()]).flatten()
    W_in = kernel[:N_IN, :]  # task-input weights (noise rows are identity*0)
    return alpha.astype(np.float64), W_in.astype(np.float64), W_rec.astype(np.float64), bias.astype(np.float64)


def build_inputs(period):
    """Pulse-driven theta->alpha->gamma sequence input, shape (T, N_IN)."""
    T = WAIT + len(MODES) * period
    u = np.zeros((T, N_IN))
    for i in range(len(MODES)):
        s = WAIT + i * period
        u[s:s + PULSE, i] = 1.0
    return u, T


def interval_index(T, period):
    """Return per-timestep interval index (0..3) or -1 during the wait period."""
    idx = np.full(T, -1)
    for i in range(len(MODES)):
        s = WAIT + i * period
        idx[s:s + period] = i
    return idx


def sim_batch(u, alpha, W_in, W_rec, bias, masks, clamp_seq):
    """Batched forward sim. masks: (B,N) 1=keep 0=silence. clamp_seq: (T,N) value
    used for silenced neurons' output at each t (0 for zero-ablation, baseline for
    freeze). Returns Z: (B, T) population-mean output (over all N, silenced=clamp)."""
    T = u.shape[0]
    B = masks.shape[0]
    X = np.zeros((B, N))
    Z = np.zeros((B, T))
    a = alpha[None, :]
    for t in range(T):
        r = np.tanh(X)
        r_eff = np.where(masks > 0, r, clamp_seq[t][None, :])
        Z[:, t] = r_eff.mean(axis=1)
        h = u[t] @ W_in + bias  # (N,)
        X = (1 - a) * X + a * (h[None, :] + r_eff @ W_rec)
    return Z


def sim_full_R(u, alpha, W_in, W_rec, bias):
    """Unablated sim returning full R (T,N) for baseline extraction."""
    T = u.shape[0]
    X = np.zeros(N)
    R = np.zeros((T, N))
    for t in range(T):
        r = np.tanh(X)
        R[t] = r
        h = u[t] @ W_in + bias
        X = (1 - alpha) * X + alpha * (h + r @ W_rec)
    return R


def band_metrics(z_interval, mode, transient=500):
    """band power ratio and absolute in-band power for one interval's z."""
    seg = z_interval[transient:]
    f, psd = sig.welch(seg, fs=FS, nperseg=min(len(seg), 4096))
    lo, hi = BANDS[mode]
    band = (f >= lo) & (f < hi)
    total = psd.sum()
    return psd[band].sum() / total if total > 0 else 0.0, psd[band].sum()


def make_masks(tau, fractions, n_random, rng):
    """Return list of (series, frac, trial, mask). slow=longest tau first."""
    order_slow = np.argsort(-tau)   # longest tau first
    order_fast = np.argsort(tau)    # shortest tau first
    out = []
    for f in fractions:
        k = int(round(f * N))
        if k == 0:
            out.append(('none', 0.0, 0, np.ones(N)))
            continue
        m = np.ones(N); m[order_slow[:k]] = 0; out.append(('slow', f, 0, m))
        m = np.ones(N); m[order_fast[:k]] = 0; out.append(('fast', f, 0, m))
        for tr in range(n_random):
            m = np.ones(N); m[rng.choice(N, k, replace=False)] = 0
            out.append(('random', f, tr, m))
    return out


def analyze_run(run_path, period, fractions, n_random, seed):
    alpha, W_in, W_rec, bias = extract_weights(run_path)
    tau = 1.0 / alpha
    u, T = build_inputs(period)
    iv = interval_index(T, period)

    # baselines per mode (unablated)
    R = sim_full_R(u, alpha, W_in, W_rec, bias)
    baseline = np.zeros((len(MODES), N))
    for i in range(len(MODES)):
        sel = np.where(iv == i)[0][500:]  # skip transient
        baseline[i] = R[sel].mean(axis=0)

    rng = np.random.default_rng(seed)
    cfgs = make_masks(tau, fractions, n_random, rng)
    masks = np.stack([c[3] for c in cfgs])

    clamp_zero = np.zeros((T, N))
    clamp_freeze = np.zeros((T, N))
    for i in range(len(MODES)):
        clamp_freeze[iv == i] = baseline[i]

    results = {}
    for method, clamp in [('zero', clamp_zero), ('freeze', clamp_freeze)]:
        Z = sim_batch(u, alpha, W_in, W_rec, bias, masks, clamp)
        for ci, (series, frac, trial, _) in enumerate(cfgs):
            for mi, mode in enumerate(MODES):
                seg = Z[ci, np.where(iv == mi)[0]]
                ratio, absp = band_metrics(seg, mode)
                results.setdefault((method, series, mode), []).append((frac, trial, ratio, absp))
    return results, tau


def aggregate(all_run_results, fractions):
    """Collapse per-run results into arrays keyed by (method, series, mode)."""
    agg = {}
    methods = ['zero', 'freeze']
    series_list = ['slow', 'fast', 'random', 'none']
    for method in methods:
        for series in series_list:
            for mode in MODES:
                # per fraction: value per run (random averaged over trials)
                per_run = {f: [] for f in fractions}
                for rr in all_run_results:
                    recs = rr.get((method, series, mode), [])
                    byf = {}
                    for frac, trial, ratio, absp in recs:
                        byf.setdefault(round(frac, 3), []).append((ratio, absp))
                    for f in fractions:
                        vals = byf.get(round(f, 3), [])
                        if vals:
                            per_run[f].append(np.mean(vals, axis=0))  # avg over trials
                agg[(method, series, mode)] = per_run
    return agg


def plot_main(agg, fractions, out_dir):
    """Main figure: zero-ablation, absolute in-band power RETENTION vs fraction."""
    for metric_idx, metric_name, fname in [(1, 'in-band power (retention)', 'ablation_abspower'),
                                           (0, 'band power ratio', 'ablation_ratio')]:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=(metric_idx == 1))
        for mi, mode in enumerate(MODES):
            ax = axes[mi]
            # f=0 reference (from 'none')
            none_run = agg[('zero', 'none', mode)][0.0]
            ref = np.mean([v[metric_idx] for v in none_run]) if none_run else 1.0
            for series in ['slow', 'fast', 'random']:
                col, mk, lab = SERIES_STYLE[series]
                means, sds, fs_ = [], [], []
                for f in fractions:
                    vals = agg[('zero', series, mode)][f] if f > 0 else none_run
                    if not vals:
                        continue
                    arr = np.array([v[metric_idx] for v in vals])
                    if metric_idx == 1:
                        arr = arr / ref  # retention
                    fs_.append(f); means.append(arr.mean()); sds.append(arr.std())
                means, sds, fs_ = np.array(means), np.array(sds), np.array(fs_)
                ax.errorbar(fs_ * 100, means, yerr=sds, color=col, marker=mk, ms=4,
                            capsize=2, lw=1.6, label=lab)
            ax.set_title(f'{mode}  ({BANDS[mode][0]}-{BANDS[mode][1]} Hz)', color=COLORS[mode])
            ax.set_xlabel('% neurons silenced')
            if mi == 0:
                ax.set_ylabel(metric_name)
            ax.grid(alpha=0.3)
            if mi == 3:
                ax.legend(fontsize=9)
        fig.suptitle(f'Zero-ablation: {metric_name} vs fraction silenced '
                     f'(mean$\\pm$SD over runs)', y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{fname}.png'), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, f'{fname}.pdf'), bbox_inches='tight')
        plt.close(fig)


def plot_zero_vs_freeze(agg, fractions, out_dir):
    """Compare zero vs freeze for slow-first ablation on beta/gamma (retention)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, mode in zip(axes, ['beta', 'gamma']):
        for method, ls in [('zero', '-'), ('freeze', '--')]:
            none_run = agg[(method, 'none', mode)][0.0]
            ref = np.mean([v[1] for v in none_run]) if none_run else 1.0
            for series in ['slow', 'fast']:
                col, mk, lab = SERIES_STYLE[series]
                fs_, means, sds = [], [], []
                for f in fractions:
                    vals = agg[(method, series, mode)][f] if f > 0 else none_run
                    if not vals:
                        continue
                    arr = np.array([v[1] for v in vals]) / ref
                    fs_.append(f); means.append(arr.mean()); sds.append(arr.std())
                ax.errorbar(np.array(fs_) * 100, means, yerr=sds, color=col, marker=mk, ms=4,
                            ls=ls, capsize=2, lw=1.5,
                            label=f'{series}-first / {method}')
        ax.set_title(f'{mode}  ({BANDS[mode][0]}-{BANDS[mode][1]} Hz)', color=COLORS[mode])
        ax.set_xlabel('% neurons silenced')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('in-band power (retention)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Zero vs Freeze ablation (beta/gamma), mean$\\pm$SD over runs', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ablation_zero_vs_freeze.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'ablation_zero_vs_freeze.pdf'), bbox_inches='tight')
    plt.close(fig)


def curve(agg, method, series, mode, fractions, metric_idx=1, normalize=True):
    """Per-run retention -> median, IQR (25,75) across runs, per fraction."""
    ref = np.array([v[metric_idx] for v in agg[(method, 'none', mode)][0.0]])
    fs, med, lo, hi = [], [], [], []
    for f in fractions:
        if f == 0:
            vals = np.ones_like(ref)
        else:
            recs = agg[(method, series, mode)][f]
            if not recs:
                continue
            v = np.array([x[metric_idx] for x in recs])
            vals = v / ref if normalize else v
        fs.append(f * 100); med.append(np.median(vals))
        lo.append(np.percentile(vals, 25)); hi.append(np.percentile(vals, 75))
    return np.array(fs), np.array(med), np.array(lo), np.array(hi)


def plot_paper_figure(agg, fractions, out_dir):
    """Publication figures.
    MAIN (fig_ablation): freeze method, 1x4 modes, median + IQR band, 3 series
    -- the clean causal test (isolates oscillatory contribution).
    SUPP (fig_ablation_zerofreeze): zero vs freeze for beta/gamma (slow vs fast)
    -- shows that full removal additionally perturbs the baseline (Mechanism B)."""
    plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10.5})
    # --- MAIN: freeze 1x4 ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ci, mode in enumerate(MODES):
        ax = axes[ci]
        for series in ['slow', 'fast', 'random']:
            col, mk, lab = SERIES_STYLE[series]
            fs, med, lo, hi = curve(agg, 'freeze', series, mode, fractions, 1, True)
            ax.fill_between(fs, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(fs, med, color=col, marker=mk, ms=4.5, lw=2.0, label=lab)
        ax.axhline(0, color='k', lw=0.6, alpha=0.3)
        ax.set_ylim(-0.08, 1.32); ax.set_xlim(-3, 92); ax.grid(alpha=0.25)
        ax.set_title(f'{mode}  ({BANDS[mode][0]}–{BANDS[mode][1]} Hz)',
                     color=COLORS[mode], fontweight='bold')
        ax.set_xlabel('% neurons silenced')
        if ci == 3:
            ax.legend(loc='upper right', framealpha=0.92)
    axes[0].set_ylabel('in-band power (retained)')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'fig_ablation.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- SUPP: zero vs freeze, beta/gamma ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, mode in zip(axes, ['beta', 'gamma']):
        for method, ls, mlab in [('freeze', 'solid', 'freeze'), ('zero', (0, (5, 2.5)), 'zero')]:
            for series in ['slow', 'fast']:
                col, mk, _ = SERIES_STYLE[series]
                fs, med, lo, hi = curve(agg, method, series, mode, fractions, 1, True)
                ax.plot(fs, med, color=col, marker=mk, ms=4, lw=1.8, ls=ls,
                        label=f'{series}-first / {mlab}')
        ax.set_title(f'{mode}  ({BANDS[mode][0]}–{BANDS[mode][1]} Hz)',
                     color=COLORS[mode], fontweight='bold')
        ax.set_xlabel('% neurons silenced'); ax.set_ylim(-0.08, 1.32); ax.grid(alpha=0.25)
    axes[0].set_ylabel('in-band power (retained, median)')
    axes[1].legend(fontsize=8.5, framealpha=0.9, handlelength=3.4, handletextpad=0.6)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'fig_ablation_zerofreeze.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_npz(agg, fractions, out_dir):
    """Save per-run values for fast replotting: key 'method|series|mode|metric' -> (n_frac, n_run)."""
    data = {'fractions': np.array(fractions)}
    for (method, series, mode), per_run in agg.items():
        for mi, met in enumerate(['ratio', 'abs']):
            n_run = max((len(per_run[f]) for f in fractions), default=0)
            arr = np.full((len(fractions), n_run), np.nan)
            for fi, f in enumerate(fractions):
                for ri, v in enumerate(per_run[f]):
                    arr[fi, ri] = v[mi]
            data[f'{method}|{series}|{mode}|{met}'] = arr
    np.savez(os.path.join(out_dir, 'ablation_raw.npz'), **data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', default='multiple_runs')
    ap.add_argument('--num_runs', type=int, default=20)
    ap.add_argument('--period', type=int, default=6000)
    ap.add_argument('--n_random', type=int, default=5)
    ap.add_argument('--out_dir', default='ablation_results')
    args = ap.parse_args()

    fractions = [round(x, 3) for x in np.arange(0.0, 0.95, 0.1)]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'fractions: {fractions}')

    all_res = []
    for rid in range(1, args.num_runs + 1):
        rp = os.path.join(args.runs_dir, str(rid))
        if not os.path.isdir(os.path.join(rp, 'results', 'checkpoints')):
            print(f'  run {rid}: no checkpoints, skip'); continue
        res, tau = analyze_run(rp, args.period, fractions, args.n_random, seed=1000 + rid)
        all_res.append(res)
        print(f'  run {rid}: done (tau {tau.min():.1f}-{tau.max():.1f} ms)')

    agg = aggregate(all_res, fractions)

    # save numeric summary (means) for beta/gamma
    summary = {}
    for (method, series, mode), per_run in agg.items():
        summary[f'{method}|{series}|{mode}'] = {
            str(f): (float(np.mean([v[0] for v in per_run[f]])) if per_run[f] else None,
                     float(np.mean([v[1] for v in per_run[f]])) if per_run[f] else None)
            for f in fractions
        }
    with open(os.path.join(args.out_dir, 'ablation_summary.json'), 'w') as fh:
        json.dump({'fractions': fractions, 'metric_order': ['band_ratio', 'abs_power'],
                   'data': summary}, fh, indent=2)

    plot_main(agg, fractions, args.out_dir)
    plot_zero_vs_freeze(agg, fractions, args.out_dir)
    plot_paper_figure(agg, fractions, args.out_dir)
    save_npz(agg, fractions, args.out_dir)
    print(f'\nSaved figures + summary to {args.out_dir}/')


if __name__ == '__main__':
    main()
