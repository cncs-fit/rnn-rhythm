#!/usr/bin/env python3
"""Generate Figure 3: Synchronisation structure.

Row A – Absolute sync strength matrices S_ij (one heatmap per mode,
        neurons sorted by τ ascending).
Row B – Sync degree d_i distributions (one histogram per mode,
        mean line annotated).

Usage:
    python make_fig3_sync.py [--run_id 1]
"""

import argparse
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from function import (load_from_json, split_signal, bandpass_filter,
                      Absolute_Sync_Strength)

# ---------- style ----------
BAND_LABELS = {"theta": r"$\theta$", "alpha": r"$\alpha$",
                "beta": r"$\beta$", "gamma": r"$\gamma$"}
BAND_COLORS = {"theta": "#E69F00", "alpha": "#CC4455",
                "beta": "#44AA88", "gamma": "#7766BB"}
TRANSIENT = 500  # ms to skip

OUT_DIR = pathlib.Path("figures")


def load_model_and_data(result_path):
    """Load model, run inference, return filtered_r_split, tau, labels."""
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
    _, z = model(inputs, noise, init_state)
    r = model.r[0].numpy().T  # (units, time)

    Fs = dsg.Fs
    wait_length = dsg.task.wait_length
    switch_num = dsg.task.switch_num
    rhythms = dsg.task.rhythms

    r_split = split_signal(r, wait_length, switch_num)

    filtered_r_split = {}
    for rhythm, (low, high) in rhythms.items():
        filtered_r_split[rhythm] = bandpass_filter(r_split, low, high, Fs)

    tau = 1.0 / np.array(alpha).flatten()

    return filtered_r_split, tau, labels, settings


def make_figure(filtered_r_split, tau, labels, settings):
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

    fig = plt.figure(figsize=(16, 8.0))
    gs = GridSpec(2, switch_num, figure=fig,
                  height_ratios=[1, 1],
                  hspace=0.45, wspace=0.30,
                  left=0.06, right=0.97, top=0.93, bottom=0.09)

    # ---- Compute sync strength matrices and degrees ----
    S_list = []
    degree_list = []
    for i, rname in enumerate(labels):
        data = filtered_r_split[rname][i]  # (units, time)
        # Skip transient
        data = data[:, TRANSIENT:]
        S = Absolute_Sync_Strength(data)
        np.fill_diagonal(S, 0)
        degree = np.sum(S, axis=1)
        S_list.append(S)
        degree_list.append(degree)

    # ---- Row A: sync strength heatmaps ----
    for i in range(switch_num):
        ax = fig.add_subplot(gs[0, i])
        rname = labels[i]
        c = BAND_COLORS[rname]
        S = S_list[i]
        vmax_i = np.max(S)

        im = ax.imshow(S, vmin=0, vmax=vmax_i, aspect="equal",
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        low, high = rhythms[rname]
        ax.set_title(f"{BAND_LABELS[rname]}  ({low}–{high} Hz)",
                     fontsize=15, color=c)

        # Colorbar
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=11)

        if i == 0:
            ax.set_ylabel("Neuron (sorted by $\\tau$)")

    # Panel A label
    fig.text(0.01, 0.96, "A", fontsize=18, fontweight="bold",
             va="top", ha="left")

    # ---- Row B: sync degree distributions ----
    # Pre-compute y-max for shared y-axis
    hist_max = 0
    for deg in degree_list:
        counts, _ = np.histogram(deg, bins=25)
        hist_max = max(hist_max, counts.max())
    hist_ylim = hist_max * 1.15

    for i in range(switch_num):
        ax = fig.add_subplot(gs[1, i])
        rname = labels[i]
        c = BAND_COLORS[rname]
        deg = degree_list[i]

        ax.hist(deg, bins=25, color=c, alpha=0.75,
                edgecolor="white", linewidth=0.4)
        ax.axvline(np.mean(deg), color="0.2", linestyle="--", linewidth=1.2)
        ax.text(np.mean(deg), hist_ylim * 0.92,
                f"  $\\bar{{d}}$={np.mean(deg):.1f}",
                fontsize=11, va="top", ha="left", color="0.2")

        low, high = rhythms[rname]
        ax.set_title(f"{BAND_LABELS[rname]}  ({low}–{high} Hz)",
                     fontsize=15, color=c)
        ax.set_xlabel("Sync degree $d_i$")
        ax.set_ylim(0, hist_ylim)

        if i == 0:
            ax.set_ylabel("Number of neurons")
        else:
            ax.set_yticklabels([])

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
    filtered_r_split, tau, labels, settings = load_model_and_data(result_path)

    fig = make_figure(filtered_r_split, tau, labels, settings)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig3_sync.pdf"
    out_png = OUT_DIR / "fig3_sync.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_pdf}, {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
