"""Check whether tau-amp correlation & amplitude concentration are stable
between the ~1e-5 checkpoint and the final ~1e-6 model, to justify relaxing
the early-stopping criterion for the R1-Major1 additional runs."""
import os, sys, json, glob
import numpy as np
from scipy import stats

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try: tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError: pass

from function import load_from_json, split_signal, bandpass_filter, calc_amplitudes

RUNS = {1: '0095', 2: '0180', 3: '0100', 5: '0090', 19: '0105'}
MODES = ['theta', 'alpha', 'beta', 'gamma']


def gini(a):
    a = np.sort(np.asarray(a).ravel())
    if a.min() < 0: return np.nan
    n = len(a); idx = np.arange(1, n + 1)
    return np.sum((2 * idx - n - 1) * a) / (n * np.sum(a))


def eval_ckpt(run, weights_file):
    rp = f'multiple_runs/{run}/results'
    model, dsg, history = load_from_json(rp)
    model.load_weights(os.path.join(rp, 'checkpoints', weights_file))
    alpha = np.asarray(model.rnn_layer.cell.alpha.numpy()).flatten()
    tau = 1.0 / alpha
    dsg.update_task_config(period_length=8000, switch_num=dsg.task.n_rhythm)
    dsg.update_noise_config(strength=0)
    inputs, noise, mask, onehots, labels = dsg.make_datasets(
        batch_size=1, specified_rhythm=dsg.task.rhythm_names)
    labels = labels[0]
    init_state = model.get_initial_state(batch_size=1)
    _ = model(inputs, noise, init_state)
    r = model.r[0].numpy().T  # (units, time)
    r_split = split_signal(r, dsg.task.wait_length, dsg.task.switch_num)
    out = {}
    for i, mode in enumerate(labels):
        low, high = dsg.task.rhythms[mode]
        filt = bandpass_filter(r_split, low, high, dsg.Fs)  # (switch,units,T)
        amp = calc_amplitudes(filt, cut=(500, -1))[i]  # matched amplitude, (units,)
        rho, _ = stats.spearmanr(tau, amp)
        out[mode] = dict(rho=rho, skew=float(stats.skew(amp)), gini=float(gini(amp)))
    return out, tau


print(f"{'run':>3} {'stage':>6} | " + " | ".join(f"{m:>21}" for m in MODES))
print(f"{'':>3} {'':>6} | " + " | ".join(f"{'rho    skew   gini':>21}" for m in MODES))
rows = {}
for run, ck5 in RUNS.items():
    for stage, wf in [('1e-5', f'weights-{ck5}.weights.h5'), ('1e-6', 'best_model.weights.h5')]:
        res, tau = eval_ckpt(run, wf)
        rows[(run, stage)] = res
        cells = " | ".join(f"{res[m]['rho']:+.2f}  {res[m]['skew']:+.2f}  {res[m]['gini']:.2f}" for m in MODES)
        print(f"{run:>3} {stage:>6} | {cells}")
    # delta
    d = rows[(run, '1e-6')]
    e = rows[(run, '1e-5')]
    drho = " | ".join(f"Δrho={d[m]['rho']-e[m]['rho']:+.3f}          " for m in MODES)
    print(f"{run:>3} {'Δ':>6} | {drho}")
    print('-' * 100)

# summary: mean |Δrho| per band across runs
print("\n=== mean |Δrho(1e-6 vs 1e-5)| across runs, per band ===")
for m in MODES:
    ds = [abs(rows[(r, '1e-6')][m]['rho'] - rows[(r, '1e-5')][m]['rho']) for r in RUNS]
    print(f"  {m:>6}: mean|Δrho|={np.mean(ds):.3f}  max|Δrho|={np.max(ds):.3f}")
