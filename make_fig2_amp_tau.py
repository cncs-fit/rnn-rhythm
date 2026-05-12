#!/usr/bin/env python3
"""Generate Figure 2: Amplitude distributions and τ–amplitude relationship.

Row A – Filtered-amplitude histograms (one panel per mode, shared y-axis).
Row B – τ vs amplitude scatter (one panel per mode, log x-axis, Spearman ρ).

Usage:
    python make_fig2_amp_tau.py [--run_id 1]
"""

import argparse
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats, signal
from function import (load_from_json, split_signal, bandpass_filter,
                      calc_amplitudes, calc_axis)

# ---------- style ----------
BAND_LABELS = {"theta": r"$\theta$", "alpha": r"$\alpha$",
                "beta": r"$\beta$", "gamma": r"$\gamma$"}
BAND_COLORS = {"theta": "#E69F00", "alpha": "#CC4455",
                "beta": "#44AA88", "gamma": "#7766BB"}
TRANSIENT = 500          # ms to skip
AMP_BINS = np.linspace(0, 2, 30)

OUT_DIR = pathlib.Path("figures")


def load_model_and_data(result_path):
    """Load model, run inference, return r_split, alpha, settings, labels."""
    model, dsg, history = load_from_json(result_path)

    ckpt_dir = os.path.join(result_path, "checkpoints")
    best_path = os.path.join(ckpt_dir, "best_model.weights.h5")
    if os.path.exists(best_path):
        model.load_weights(best_path)
    else:
        ckpt = [f for f in os.listdir(ckpt_dir)
                if f.startswith("best") and ".weights.h5" in f][0]
        model.load_weights(os.path.join(ckpt_dir, ckpt))

    try:
        model.sort_by_tau()
    except Exception:
        pass

    # Get alpha (leak rate)
    try:
        alpha = model.rnn_layer.cell.alpha.numpy()
    except Exception:
        alpha = model.rnn_layer.cell.alpha

    dsg.update_task_config(period_length=8000,
                           switch_num=dsg.task.n_rhythm)
    dsg.update_noise_config(strength=0)

    settings = dsg.get_config()

    inputs, noise, mask, onehots, labels = dsg.make_datasets(
        batch_size=1, specified_rhythm=dsg.task.rhythm_names)
    labels = labels[0]

    init_state = model.get_initial_state(batch_size=1)
    y, z = model(inputs, noise, init_state)
    r = model.r[0].numpy().T  # (units, time)

    Fs = dsg.Fs
    wait_length = dsg.task.wait_length
    switch_num = dsg.task.switch_num
    rhythms = dsg.task.rhythms

    # Split by mode
    r_split = split_signal(r, wait_length, switch_num)

    # Band-pass filter for each rhythm
    filtered_r_split = {}
    for rhythm, (low, high) in rhythms.items():
        filtered_r_split[rhythm] = bandpass_filter(r_split, low, high, Fs)

    # Filtered amplitudes: for each mode, Hilbert envelope mean in the
    # corresponding mode interval
    filtered_amps = {rhythm: calc_amplitudes(filtered_r_split[rhythm],
                                             cut=(TRANSIENT, -1))
                     for rhythm in dsg.task.rhythm_names}

    tau = 1.0 / np.array(alpha).flatten()

    return filtered_amps, tau, labels, settings


def make_figure(filtered_amps, tau, labels, settings):
    rhythms = settings["task_config"]["rhythms"]
    switch_num = len(labels)

    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "font.size": 14, "axes.labelsize": 16,
        "axes.titlesize": 16, "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    fig = plt.figure(figsize=(16, 7.5))
    gs = GridSpec(2, switch_num, figure=fig,
                  hspace=0.45, wspace=0.25,
                  left=0.06, right=0.97, top=0.93, bottom=0.09)

    # ---- Row A: amplitude histograms ----
    # Pre-compute y-max for shared y-axis
    hist_max = 0
    for i, rname in enumerate(labels):
        counts, _ = np.histogram(filtered_amps[rname][i], bins=AMP_BINS)
        hist_max = max(hist_max, counts.max())
    hist_ylim = hist_max * 1.15

    ax_hist = []
    for i in range(switch_num):
        ax = fig.add_subplot(gs[0, i])
        rname = labels[i]
        c = BAND_COLORS[rname]

        ax.hist(filtered_amps[rname][i], bins=AMP_BINS,
                color=c, alpha=0.75, edgecolor="white", linewidth=0.4)

        low, high = rhythms[rname]
        ax.set_title(f"{BAND_LABELS[rname]}  ({low}–{high} Hz)",
                     fontsize=15, color=c)
        ax.set_xlabel("Amplitude")
        ax.set_ylim(0, hist_ylim)

        if i == 0:
            ax.set_ylabel("Number of neurons")
        else:
            ax.set_yticklabels([])

        ax_hist.append(ax)

    # Panel A label
    fig.text(0.01, 0.96, "A", fontsize=18, fontweight="bold",
             va="top", ha="left")

    # ---- Row B: τ–amplitude scatter ----
    # Pre-compute common amplitude range across modes
    amp_max = max(np.max(filtered_amps[labels[i]][i]) for i in range(switch_num))

    for i in range(switch_num):
        ax = fig.add_subplot(gs[1, i])
        rname = labels[i]
        c = BAND_COLORS[rname]
        amp = filtered_amps[rname][i]

        ax.scatter(tau, amp, color=c, alpha=0.6, s=18, edgecolors="none")

        ax.set_xscale("log")
        ticks = [10, 20, 50, 100, 200]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel(r"Time constant $\tau$ [ms]")
        ax.set_ylim(bottom=0, top=amp_max * 1.08)

        if i == 0:
            ax.set_ylabel("Amplitude")
        else:
            ax.set_yticklabels([])

        # Spearman correlation
        r_s, _ = stats.spearmanr(tau, amp)
        ax.text(0.95, 0.95,
                f"$\\rho_s$ = {r_s:.2f}",
                transform=ax.transAxes, fontsize=12,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.8, edgecolor="0.7"))

        low, high = rhythms[rname]
        ax.set_title(f"{BAND_LABELS[rname]}  ({low}–{high} Hz)",
                     fontsize=15, color=c)

    # Panel B label
    fig.text(0.01, 0.49, "B", fontsize=18, fontweight="bold",
             va="top", ha="left")

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=1,
                        help="Run ID under multiple_runs/ (default: 1)")
    args = parser.parse_args()

    result_path = f"multiple_runs/{args.run_id}/results"
    print(f"Loading from {result_path} ...")
    filtered_amps, tau, labels, settings = load_model_and_data(result_path)

    fig = make_figure(filtered_amps, tau, labels, settings)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig2_amp_tau.pdf"
    out_png = OUT_DIR / "fig2_amp_tau.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_pdf}, {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
