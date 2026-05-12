#!/usr/bin/env python3
"""Generate Figure 1 B, C: Task overview.

(B) Collective output z(t) time series with mode labels.
(C) Power spectra for each rhythm segment.

Panel (A) (schematic) is created separately.

Usage:
    python make_fig1_task_overview.py [--run_id 1]
"""

import argparse
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from function import load_from_json, split_signal, calc_axis, power

# ---------- config ----------
BAND_LABELS = {"theta": r"$\theta$", "alpha": r"$\alpha$",
                "beta": r"$\beta$", "gamma": r"$\gamma$"}
BAND_COLORS = {"theta": "#E69F00", "alpha": "#CC4455",
                "beta": "#44AA88", "gamma": "#7766BB"}
# transient to skip in power spectrum (ms)
TRANSIENT = 500
# time window to show in zoomed waveform (ms)
ZOOM_WINDOW = 2000

OUT_DIR = pathlib.Path("figures")


def load_model_and_data(result_path):
    """Load model, run inference with noise=0, return z, settings."""
    model, dsg, history = load_from_json(result_path)

    # Load best weights
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

    # Task config for full-length segments, no noise
    dsg.update_task_config(period_length=8000,
                           switch_num=dsg.task.n_rhythm)
    dsg.update_noise_config(strength=0)

    inputs, noise, mask, onehots, labels = dsg.make_datasets(
        batch_size=1, specified_rhythm=dsg.task.rhythm_names)
    labels = labels[0]

    init_state = model.get_initial_state(batch_size=1)
    _, z = model(inputs, noise, init_state)
    z = z[0, :, 0].numpy() if hasattr(z[0, :, 0], 'numpy') else np.array(z[0, :, 0])

    settings = dsg.get_config()
    return z, inputs, labels, settings


def make_figure(z, inputs, labels, settings):
    Fs = settings["Fs"]
    task = settings["task_config"]
    wait_length = task["wait_length"]
    period_length = task["period_length"]
    switch_num = task["switch_num"]
    rhythms = task["rhythms"]
    data_length = settings["data_length"]

    t_ax, f_ax = calc_axis(data_length, period_length, Fs)

    # Split output by mode
    z_split = split_signal(z, wait_length, switch_num)
    # Power spectrum (skip transient)
    z_trimmed = z_split[..., TRANSIENT:]
    pz = power(z_trimmed, Fs).numpy() if hasattr(power(z_trimmed, Fs), 'numpy') else np.array(power(z_trimmed, Fs))
    f_ax_trimmed = np.arange(1, 1 + z_trimmed.shape[-1] // 2, dtype=np.float32) * (Fs / z_trimmed.shape[-1])

    # Peak frequency per segment
    max_Hz = f_ax_trimmed[np.argmax(pz, axis=-1)]

    # ---- Layout ----
    plt.rcParams.update({"font.family": "Liberation Sans",
                         "mathtext.fontset": "custom",
                         "mathtext.rm": "Liberation Sans",
                         "mathtext.it": "Liberation Sans:italic",
                         "mathtext.bf": "Liberation Sans:bold",
                         "font.size": 14, "axes.labelsize": 16,
                         "axes.titlesize": 16, "xtick.labelsize": 13,
                         "ytick.labelsize": 13})
    fig = plt.figure(figsize=(14, 5.5))
    gs = GridSpec(2, switch_num, figure=fig,
                  height_ratios=[1.2, 1],
                  hspace=0.42, wspace=0.30,
                  left=0.06, right=0.97, top=0.91, bottom=0.12)

    # ---- Panel B: full time series ----
    ax_ts = fig.add_subplot(gs[0, :])

    ax_ts.plot(t_ax, z, color="0.2", linewidth=0.4)

    # Mode shading + labels
    rhythm_names = list(rhythms.keys())
    for i in range(switch_num):
        t_start = (wait_length + i * period_length) / Fs
        t_end = (wait_length + (i + 1) * period_length) / Fs
        rname = labels[i]
        c = BAND_COLORS.get(rname, "gray")
        ax_ts.axvspan(t_start, t_end, alpha=0.10, color=c, zorder=0)
        ax_ts.text((t_start + t_end) / 2, ax_ts.get_ylim()[1] if i > 0 else np.max(z) * 1.02,
                   BAND_LABELS.get(rname, rname),
                   ha="center", va="bottom", fontsize=18, color=c, fontweight="bold")

    # Wait period shading
    if wait_length > 0:
        ax_ts.axvspan(0, wait_length / Fs, alpha=0.06, color="gray", zorder=0)

    # Mode boundaries
    for i in range(switch_num + 1):
        ax_ts.axvline((wait_length + i * period_length) / Fs,
                      color="0.5", linewidth=0.6, linestyle="--", zorder=1)

    ax_ts.set_xlabel("Time [s]")
    ax_ts.set_ylabel("$z(t)$")
    ax_ts.set_xlim(t_ax[0], t_ax[-1])
    ax_ts.set_title("B", fontsize=18, loc="left", fontweight="bold")

    # ---- Panel C: power spectra ----
    # Common y limits
    pz_max = np.max(pz[:, f_ax_trimmed <= 100])
    pz_min = np.min(pz[:, (f_ax_trimmed >= 1) & (f_ax_trimmed <= 100)])
    pz_min = max(pz_min, pz_max * 1e-12)  # avoid too large range

    for i in range(switch_num):
        ax = fig.add_subplot(gs[1, i])
        rname = labels[i]
        c = BAND_COLORS.get(rname, "tab:blue")

        # Band shading for all bands
        for bname, (low, high) in rhythms.items():
            alpha_fill = 0.25 if bname == rname else 0.06
            ax.axvspan(low, high, alpha=alpha_fill,
                       color=BAND_COLORS.get(bname, "gray"), zorder=0)

        ax.plot(f_ax_trimmed, pz[i], color=c, linewidth=1.0)
        ax.set_yscale("log")
        ax.set_xlim(0, 80)
        ax.set_ylim(pz_min * 0.3, pz_max * 5)
        ax.set_xlabel("Frequency [Hz]")

        # Peak annotation
        ax.annotate(f"peak {max_Hz[i]:.1f} Hz",
                    xy=(max_Hz[i], pz[i, np.argmax(pz[i])]),
                    xytext=(max_Hz[i] + 8, pz[i, np.argmax(pz[i])] * 0.3),
                    fontsize=11, color=c,
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.8))

        # Title with band name
        low, high = rhythms[rname]
        ax.set_title(f"{BAND_LABELS[rname]}  ({low}–{high} Hz)",
                     fontsize=15, color=c)

        if i == 0:
            ax.set_ylabel("Power")
        else:
            ax.set_yticklabels([])

    # Overall panel C label
    fig.text(0.06, 0.48, "C", fontsize=18, fontweight="bold",
             va="top", ha="left")

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=1,
                        help="Run ID under multiple_runs/ (default: 1)")
    args = parser.parse_args()

    result_path = f"multiple_runs/{args.run_id}/results"
    print(f"Loading from {result_path} ...")
    z, inputs, labels, settings = load_model_and_data(result_path)

    fig = make_figure(z, inputs, labels, settings)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig1_task_overview.pdf"
    out_png = OUT_DIR / "fig1_task_overview.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_pdf}, {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
