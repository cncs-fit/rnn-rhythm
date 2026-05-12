"""
analysis_multiple.py
====================
Aggregate the results of multiple training runs (collected under
``multiple_runs/``), compute trial-wise means and standard deviations of each
metric, and write summary plots and CSV tables.

Usage:
    python analysis_multiple.py [--runs_dir multiple_runs] [--num_runs 5]
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal

from function import (
    load_from_json, split_signal, calc_amplitudes,
    bandpass_filter, hilbert, SPLV, Absolute_Sync_Strength, calc_axis,
)

# ---------- Configuration ----------
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 14
COLOR = {
    'theta': 'orange', 'alpha': 'indianred',
    'beta': 'mediumaquamarine', 'gamma': 'mediumpurple',
}
MODE_ORDER = ['theta', 'alpha', 'beta', 'gamma']

# ---------- Utilities ----------

def gini(array):
    """Gini coefficient."""
    a = np.sort(np.asarray(array).ravel())
    if np.amin(a) < 0:
        return np.nan
    n = len(a)
    idx = np.arange(1, n + 1)
    return (np.sum((2 * idx - n - 1) * a)) / (n * np.sum(a))


def load_run(run_path):
    """Restore one training run, run inference, and compute all per-run metrics."""
    result_path = os.path.join(run_path, 'results')

    # --- Load the trained model and the dataset configuration ---
    model, dsg, history = load_from_json(result_path)

    weights_path = os.path.join(result_path, 'checkpoints', 'best_model.weights.h5')
    if not os.path.exists(weights_path):
        ckpts = [f for f in os.listdir(os.path.join(result_path, 'checkpoints'))
                 if f.startswith('best') and '.weights.h5' in f]
        if ckpts:
            weights_path = os.path.join(result_path, 'checkpoints', ckpts[0])
        else:
            weights_path = os.path.join(result_path, 'checkpoints', 'last_weight.weights.h5')
    model.load_weights(weights_path)

    # sort by tau
    try:
        model.sort_by_tau()
        alpha = model.rnn_layer.cell.alpha.numpy()
    except Exception:
        alpha = model.rnn_layer.cell.alpha
    tau = 1.0 / np.asarray(alpha).flatten()

    # --- Run a noise-free simulation ---
    dsg.update_task_config(period_length=8000, switch_num=dsg.task.n_rhythm)
    dsg.update_noise_config(strength=0)

    inputs, noise, mask, onehots, labels = dsg.make_datasets(
        batch_size=1, specified_rhythm=dsg.task.rhythm_names,
    )
    labels = labels[0]
    init_state = model.get_initial_state(batch_size=1)
    y, z = model(inputs, noise, init_state)
    r = model.r[0].numpy().T  # (units, time)

    Fs = dsg.Fs
    wait_length = dsg.task.wait_length
    switch_num = dsg.task.switch_num
    period_length = dsg.task.period_length
    rhythms = dsg.task.rhythms

    # split & filter
    r_split = split_signal(r, wait_length, switch_num)
    filtered_r_split = {}
    for rhythm, (low, high) in rhythms.items():
        filtered_r_split[rhythm] = bandpass_filter(r_split, low, high, Fs)

    # ---------- Metric computation ----------
    metrics = {}

    # (0) Minimum loss
    metrics['min_loss'] = min(history['loss'])

    # (1) Amplitude statistics (filtered): skewness, kurtosis, Gini coefficient
    filtered_amps = {
        rhythm: calc_amplitudes(filtered_r_split[rhythm], cut=(500, -1))
        for rhythm in dsg.task.rhythm_names
    }
    amp_stats = []
    for i, mode in enumerate(labels):
        data = filtered_amps[mode][i]
        s = stats.skew(data)
        k = stats.kurtosis(data)
        g = gini(data)
        _, p_s = stats.skewtest(data)
        _, p_k = stats.kurtosistest(data)
        amp_stats.append({
            'mode': mode, 'skewness': s, 'p_skewness': p_s,
            'kurtosis': k, 'p_kurtosis': p_k, 'gini': g,
        })
    metrics['amp_stats'] = amp_stats

    # (2) Time-constant vs amplitude correlation (Spearman)
    tau_amp_corr = []
    for i, mode in enumerate(labels):
        amp = filtered_amps[mode][i]
        r_s, p_s = stats.spearmanr(tau, amp)
        tau_amp_corr.append({'mode': mode, 'r_spearman': r_s, 'p_spearman': p_s})
    metrics['tau_amp_corr'] = tau_amp_corr

    # (3) Cross-mode correlation matrix of matched-mode amplitudes
    dominant_amp = {}
    for i, mode in enumerate(labels):
        dominant_amp[mode] = filtered_amps[mode][i]
    corr_matrix = pd.DataFrame(dominant_amp).corr().values  # 4×4
    metrics['dominant_amp_corr_matrix'] = corr_matrix
    metrics['dominant_amp_corr_labels'] = list(labels)

    # (4) Selectivity Index (Power)
    filtered_powers = {}
    for rhythm in dsg.task.rhythm_names:
        sig = filtered_r_split[rhythm][..., 500:-1]
        filtered_powers[rhythm] = np.mean(sig ** 2, axis=-1)

    si_power_stats = []
    for i, mode in enumerate(labels):
        target = filtered_powers[mode][i]
        others = np.mean([filtered_powers[mode][j]
                          for j in range(len(labels)) if j != i], axis=0)
        denom = target + others
        denom[denom == 0] = 1e-10
        si = (target - others) / denom
        si_power_stats.append({
            'mode': mode, 'mean_si': np.mean(si), 'std_si': np.std(si),
            'positive_ratio': np.sum(si > 0) / len(si),
        })
    metrics['si_power'] = si_power_stats

    # (5) Selectivity Index (Amplitude)
    si_amp_stats = []
    for i, mode in enumerate(labels):
        target = filtered_amps[mode][i]
        others = np.mean([filtered_amps[mode][j]
                          for j in range(len(labels)) if j != i], axis=0)
        denom = target + others
        denom[denom == 0] = 1e-10
        si = (target - others) / denom
        si_amp_stats.append({
            'mode': mode, 'mean_si': np.mean(si), 'std_si': np.std(si),
            'positive_ratio': np.sum(si > 0) / len(si),
        })
    metrics['si_amp'] = si_amp_stats

    # (6) Sync-degree statistics (from Absolute Sync Strength)
    sync_degree_stats = []
    sync_degrees = {}
    for i, mode in enumerate(labels):
        data_f = filtered_r_split[mode][i]
        S = Absolute_Sync_Strength(data_f)
        np.fill_diagonal(S, 0)
        degree = np.sum(S, axis=1)
        sync_degrees[mode] = degree

        s_val = stats.skew(degree)
        k_val = stats.kurtosis(degree)
        g_val = gini(degree)
        sync_degree_stats.append({
            'mode': mode, 'mean_deg': np.mean(degree), 'std_deg': np.std(degree),
            'skewness': s_val, 'kurtosis': k_val, 'gini': g_val,
        })
    metrics['sync_degree_stats'] = sync_degree_stats

    # (7) Sync degree vs time-constant correlation (Spearman)
    sync_tau_corr = []
    for mode in labels:
        r_s, p_s = stats.spearmanr(tau, sync_degrees[mode])
        sync_tau_corr.append({'mode': mode, 'r_spearman': r_s, 'p_spearman': p_s})
    metrics['sync_tau_corr'] = sync_tau_corr

    # (7b) Cross-mode correlation matrix of sync degree
    sync_degree_corr_matrix = pd.DataFrame(
        {mode: sync_degrees[mode] for mode in labels}
    ).corr().values  # 4×4
    metrics['sync_degree_corr_matrix'] = sync_degree_corr_matrix
    metrics['sync_degree_corr_labels'] = list(labels)

    # (8) Active group overlap (Jaccard)
    top_percent = 20
    n_units = len(tau)
    n_top = int(n_units * top_percent / 100)
    active_sets = {}
    for mode in labels:
        top_idx = np.argsort(sync_degrees[mode])[-n_top:]
        active_sets[mode] = set(top_idx)

    n_modes = len(labels)
    jaccard_matrix = np.zeros((n_modes, n_modes))
    for i, m1 in enumerate(labels):
        for j, m2 in enumerate(labels):
            inter = len(active_sets[m1] & active_sets[m2])
            union = len(active_sets[m1] | active_sets[m2])
            jaccard_matrix[i, j] = inter / union if union > 0 else 0
    metrics['jaccard_matrix'] = jaccard_matrix
    metrics['jaccard_labels'] = list(labels)

    return metrics


# ---------- Aggregation and plotting ----------

def aggregate_and_plot(all_metrics, out_dir):
    """Aggregate per-run metrics and write summary tables and figures."""
    os.makedirs(out_dir, exist_ok=True)
    n_runs = len(all_metrics)
    modes = MODE_ORDER  # fixed display order

    # ============================
    # (0) Loss summary
    # ============================
    losses = [m['min_loss'] for m in all_metrics]
    summary_rows = [{'metric': 'min_loss', 'mean': np.mean(losses),
                     'std': np.std(losses), 'values': str(losses)}]

    # ============================
    # (1) Amplitude statistics (skewness, kurtosis, Gini) -- mean +/- SD
    # ============================
    amp_stat_keys = ['skewness', 'kurtosis', 'gini']
    amp_agg = {mode: {k: [] for k in amp_stat_keys} for mode in modes}
    for m in all_metrics:
        for entry in m['amp_stats']:
            mode = entry['mode']
            if mode in amp_agg:
                for k in amp_stat_keys:
                    amp_agg[mode][k].append(entry[k])

    rows_amp = []
    for mode in modes:
        row = {'mode': mode}
        for k in amp_stat_keys:
            vals = amp_agg[mode][k]
            row[f'{k}_mean'] = np.mean(vals)
            row[f'{k}_std'] = np.std(vals)
        rows_amp.append(row)
    df_amp = pd.DataFrame(rows_amp)
    df_amp.to_csv(os.path.join(out_dir, 'amp_stats_summary.csv'), index=False)

    # Bar plot (skewness, kurtosis, Gini)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, k in zip(axes, amp_stat_keys):
        means = [np.mean(amp_agg[mode][k]) for mode in modes]
        stds = [np.std(amp_agg[mode][k]) for mode in modes]
        colors = [COLOR[m] for m in modes]
        ax.bar(modes, means, yerr=stds, color=colors, capsize=5, alpha=0.8, edgecolor='black')
        ax.set_title(k.capitalize())
        ax.set_ylabel(k)
        # Overlay individual data points
        for i, mode in enumerate(modes):
            ax.scatter([i] * len(amp_agg[mode][k]), amp_agg[mode][k],
                       color='black', zorder=5, s=20, alpha=0.6)
    fig.suptitle(f'Amplitude Distribution Statistics (n={n_runs})', fontsize=16)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'amp_stats_bar.png'), dpi=300)
    fig.savefig(os.path.join(out_dir, 'amp_stats_bar.pdf'))
    plt.close(fig)

    # ============================
    # (2) Time-constant vs amplitude correlation (Spearman)
    # ============================
    tau_corr_agg = {mode: {'r': [], 'p': []} for mode in modes}
    for m in all_metrics:
        for entry in m['tau_amp_corr']:
            mode = entry['mode']
            if mode in tau_corr_agg:
                tau_corr_agg[mode]['r'].append(entry['r_spearman'])
                tau_corr_agg[mode]['p'].append(entry['p_spearman'])

    rows_tc = []
    for mode in modes:
        rs = tau_corr_agg[mode]['r']
        ps = tau_corr_agg[mode]['p']
        rows_tc.append({
            'mode': mode,
            'r_spearman_mean': np.mean(rs), 'r_spearman_std': np.std(rs),
            'p_spearman_mean': np.mean(ps), 'p_spearman_std': np.std(ps),
        })
    df_tc = pd.DataFrame(rows_tc)
    df_tc.to_csv(os.path.join(out_dir, 'tau_amp_corr_summary.csv'), index=False)

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(tau_corr_agg[m]['r']) for m in modes]
    stds = [np.std(tau_corr_agg[m]['r']) for m in modes]
    colors = [COLOR[m] for m in modes]
    bars = ax.bar(modes, means, yerr=stds, color=colors, capsize=5, alpha=0.8, edgecolor='black')
    for i, mode in enumerate(modes):
        ax.scatter([i] * len(tau_corr_agg[mode]['r']), tau_corr_agg[mode]['r'],
                   color='black', zorder=5, s=20, alpha=0.6)
    ax.set_ylabel('Spearman ρ')
    ax.set_title(f'Tau–Amplitude Correlation (n={n_runs})')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'tau_amp_corr_bar.png'), dpi=300)
    fig.savefig(os.path.join(out_dir, 'tau_amp_corr_bar.pdf'))
    plt.close(fig)

    # ============================
    # (3) Cross-mode amplitude correlation matrix -- mean
    # ============================
    corr_matrices = np.stack([m['dominant_amp_corr_matrix'] for m in all_metrics])
    mean_corr = np.mean(corr_matrices, axis=0)
    std_corr = np.std(corr_matrices, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # mean
    im0 = axes[0].imshow(mean_corr, vmin=-1, vmax=1, cmap='coolwarm')
    axes[0].set_xticks(range(4)); axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(modes); axes[0].set_yticklabels(modes)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f'{mean_corr[i, j]:.2f}', ha='center', va='center', fontsize=11)
    axes[0].set_title('Mean Correlation')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # std
    im1 = axes[1].imshow(std_corr, vmin=0, cmap='Greys')
    axes[1].set_xticks(range(4)); axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(modes); axes[1].set_yticklabels(modes)
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f'{std_corr[i, j]:.3f}', ha='center', va='center', fontsize=11)
    axes[1].set_title('Std of Correlation')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle(f'Dominant Amplitude Correlation Matrix (n={n_runs})', fontsize=15)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'dominant_amp_corr_summary.png'), dpi=300)
    fig.savefig(os.path.join(out_dir, 'dominant_amp_corr_summary.pdf'))
    plt.close(fig)

    # CSV
    df_corr_mean = pd.DataFrame(mean_corr, index=modes, columns=modes)
    df_corr_mean.to_csv(os.path.join(out_dir, 'dominant_amp_corr_mean.csv'))
    df_corr_std = pd.DataFrame(std_corr, index=modes, columns=modes)
    df_corr_std.to_csv(os.path.join(out_dir, 'dominant_amp_corr_std.csv'))

    # ============================
    # (4) Selectivity Index (Power & Amplitude)
    # ============================
    for si_key, si_label in [('si_power', 'Power'), ('si_amp', 'Amplitude')]:
        si_agg = {mode: {'mean_si': [], 'positive_ratio': []} for mode in modes}
        for m in all_metrics:
            for entry in m[si_key]:
                mode = entry['mode']
                if mode in si_agg:
                    si_agg[mode]['mean_si'].append(entry['mean_si'])
                    si_agg[mode]['positive_ratio'].append(entry['positive_ratio'])

        rows_si = []
        for mode in modes:
            rows_si.append({
                'mode': mode,
                'mean_SI_mean': np.mean(si_agg[mode]['mean_si']),
                'mean_SI_std': np.std(si_agg[mode]['mean_si']),
                'positive_ratio_mean': np.mean(si_agg[mode]['positive_ratio']),
                'positive_ratio_std': np.std(si_agg[mode]['positive_ratio']),
            })
        df_si = pd.DataFrame(rows_si)
        df_si.to_csv(os.path.join(out_dir, f'selectivity_index_{si_label.lower()}_summary.csv'), index=False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Mean SI
        means = [np.mean(si_agg[m]['mean_si']) for m in modes]
        stds_ = [np.std(si_agg[m]['mean_si']) for m in modes]
        colors_ = [COLOR[m] for m in modes]
        axes[0].bar(modes, means, yerr=stds_, color=colors_, capsize=5, alpha=0.8, edgecolor='black')
        for i, mode in enumerate(modes):
            axes[0].scatter([i] * len(si_agg[mode]['mean_si']), si_agg[mode]['mean_si'],
                            color='black', zorder=5, s=20, alpha=0.6)
        axes[0].set_ylabel('Mean SI')
        axes[0].set_title(f'Selectivity Index ({si_label})')
        axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)

        # Positive ratio
        means_p = [np.mean(si_agg[m]['positive_ratio']) for m in modes]
        stds_p = [np.std(si_agg[m]['positive_ratio']) for m in modes]
        axes[1].bar(modes, means_p, yerr=stds_p, color=colors_, capsize=5, alpha=0.8, edgecolor='black')
        for i, mode in enumerate(modes):
            axes[1].scatter([i] * len(si_agg[mode]['positive_ratio']),
                            si_agg[mode]['positive_ratio'],
                            color='black', zorder=5, s=20, alpha=0.6)
        axes[1].set_ylabel('Positive Ratio')
        axes[1].set_title(f'SI > 0 Ratio ({si_label})')
        axes[1].set_ylim(0, 1.05)

        fig.suptitle(f'Selectivity Index – {si_label} (n={n_runs})', fontsize=15)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'selectivity_index_{si_label.lower()}_bar.png'), dpi=300)
        fig.savefig(os.path.join(out_dir, f'selectivity_index_{si_label.lower()}_bar.pdf'))
        plt.close(fig)

    # ============================
    # (5) Sync-degree statistics
    # ============================
    sd_keys = ['mean_deg', 'std_deg', 'skewness', 'kurtosis', 'gini']
    sd_agg = {mode: {k: [] for k in sd_keys} for mode in modes}
    for m in all_metrics:
        for entry in m['sync_degree_stats']:
            mode = entry['mode']
            if mode in sd_agg:
                for k in sd_keys:
                    sd_agg[mode][k].append(entry[k])

    rows_sd = []
    for mode in modes:
        row = {'mode': mode}
        for k in sd_keys:
            vals = sd_agg[mode][k]
            row[f'{k}_mean'] = np.mean(vals)
            row[f'{k}_std'] = np.std(vals)
        rows_sd.append(row)
    df_sd = pd.DataFrame(rows_sd)
    df_sd.to_csv(os.path.join(out_dir, 'sync_degree_stats_summary.csv'), index=False)

    fig, axes = plt.subplots(1, len(sd_keys), figsize=(4 * len(sd_keys), 5))
    for ax, k in zip(axes, sd_keys):
        means = [np.mean(sd_agg[mode][k]) for mode in modes]
        stds_ = [np.std(sd_agg[mode][k]) for mode in modes]
        colors_ = [COLOR[m] for m in modes]
        ax.bar(modes, means, yerr=stds_, color=colors_, capsize=5, alpha=0.8, edgecolor='black')
        for i, mode in enumerate(modes):
            ax.scatter([i] * len(sd_agg[mode][k]), sd_agg[mode][k],
                       color='black', zorder=5, s=20, alpha=0.6)
        ax.set_title(k)
    fig.suptitle(f'Sync Degree Statistics (n={n_runs})', fontsize=15)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sync_degree_stats_bar.png'), dpi=300)
    fig.savefig(os.path.join(out_dir, 'sync_degree_stats_bar.pdf'))
    plt.close(fig)

    # ============================
    # (6) Sync-degree vs time-constant correlation
    # ============================
    st_agg = {mode: {'r': [], 'p': []} for mode in modes}
    for m in all_metrics:
        for entry in m['sync_tau_corr']:
            mode = entry['mode']
            if mode in st_agg:
                st_agg[mode]['r'].append(entry['r_spearman'])
                st_agg[mode]['p'].append(entry['p_spearman'])

    rows_st = []
    for mode in modes:
        rs = st_agg[mode]['r']
        rows_st.append({
            'mode': mode,
            'r_spearman_mean': np.mean(rs), 'r_spearman_std': np.std(rs),
        })
    df_st = pd.DataFrame(rows_st)
    df_st.to_csv(os.path.join(out_dir, 'sync_tau_corr_summary.csv'), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(st_agg[m]['r']) for m in modes]
    stds_ = [np.std(st_agg[m]['r']) for m in modes]
    colors_ = [COLOR[m] for m in modes]
    ax.bar(modes, means, yerr=stds_, color=colors_, capsize=5, alpha=0.8, edgecolor='black')
    for i, mode in enumerate(modes):
        ax.scatter([i] * len(st_agg[mode]['r']), st_agg[mode]['r'],
                   color='black', zorder=5, s=20, alpha=0.6)
    ax.set_ylabel('Spearman ρ')
    ax.set_title(f'Sync Degree–Tau Correlation (n={n_runs})')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sync_tau_corr_bar.png'), dpi=300)
    fig.savefig(os.path.join(out_dir, 'sync_tau_corr_bar.pdf'))
    plt.close(fig)

    # ============================
    # (6b) Cross-mode sync-degree correlation matrix -- mean
    # ============================
    if 'sync_degree_corr_matrix' in all_metrics[0]:
        sd_corr_matrices = np.stack([m['sync_degree_corr_matrix'] for m in all_metrics])
        mean_sd_corr = np.mean(sd_corr_matrices, axis=0)
        std_sd_corr = np.std(sd_corr_matrices, axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # mean
        im0 = axes[0].imshow(mean_sd_corr, vmin=-1, vmax=1, cmap='coolwarm')
        axes[0].set_xticks(range(4)); axes[0].set_yticks(range(4))
        axes[0].set_xticklabels(modes); axes[0].set_yticklabels(modes)
        for i in range(4):
            for j in range(4):
                axes[0].text(j, i, f'{mean_sd_corr[i, j]:.2f}', ha='center', va='center', fontsize=11)
        axes[0].set_title('Mean Correlation')
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        # std
        im1 = axes[1].imshow(std_sd_corr, vmin=0, cmap='Greys')
        axes[1].set_xticks(range(4)); axes[1].set_yticks(range(4))
        axes[1].set_xticklabels(modes); axes[1].set_yticklabels(modes)
        for i in range(4):
            for j in range(4):
                axes[1].text(j, i, f'{std_sd_corr[i, j]:.3f}', ha='center', va='center', fontsize=11)
        axes[1].set_title('Std of Correlation')
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        fig.suptitle(f'Sync Degree Inter-Mode Correlation Matrix (n={n_runs})', fontsize=15)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, 'sync_degree_corr_summary.png'), dpi=300)
        fig.savefig(os.path.join(out_dir, 'sync_degree_corr_summary.pdf'))
        plt.close(fig)

        # CSV
        df_sd_corr_mean = pd.DataFrame(mean_sd_corr, index=modes, columns=modes)
        df_sd_corr_mean.to_csv(os.path.join(out_dir, 'sync_degree_corr_mean.csv'))

    # ============================
    # (7) Active group overlap (Jaccard) -- mean
    # ============================
    jac_matrices = np.stack([m['jaccard_matrix'] for m in all_metrics])
    mean_jac = np.mean(jac_matrices, axis=0)
    std_jac = np.std(jac_matrices, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].imshow(mean_jac, vmin=0, vmax=1, cmap='YlOrRd')
    axes[0].set_xticks(range(4)); axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(modes); axes[0].set_yticklabels(modes)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f'{mean_jac[i, j]:.2f}', ha='center', va='center', fontsize=11)
    axes[0].set_title('Mean Jaccard')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(std_jac, vmin=0, cmap='Greys')
    axes[1].set_xticks(range(4)); axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(modes); axes[1].set_yticklabels(modes)
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f'{std_jac[i, j]:.3f}', ha='center', va='center', fontsize=11)
    axes[1].set_title('Std of Jaccard')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle(f'Active Group Overlap – Jaccard (Top 20%, n={n_runs})', fontsize=15)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'jaccard_summary.png'), dpi=300)
    fig.savefig(os.path.join(out_dir, 'jaccard_summary.pdf'))
    plt.close(fig)

    df_jac = pd.DataFrame(mean_jac, index=modes, columns=modes)
    df_jac.to_csv(os.path.join(out_dir, 'jaccard_mean.csv'))
    df_jac_std = pd.DataFrame(std_jac, index=modes, columns=modes)
    df_jac_std.to_csv(os.path.join(out_dir, 'jaccard_std.csv'))

    # ============================
    # (8) Overall summary table (LaTeX-friendly)
    # ============================
    all_rows = []
    # loss
    all_rows.append({'category': 'Training', 'metric': 'Min Loss',
                     'mean': np.mean(losses), 'std': np.std(losses)})
    # amp stats
    for mode in modes:
        for k in amp_stat_keys:
            vals = amp_agg[mode][k]
            all_rows.append({
                'category': f'Amp ({mode})', 'metric': k,
                'mean': np.mean(vals), 'std': np.std(vals),
            })
    # tau-amp corr
    for mode in modes:
        rs = tau_corr_agg[mode]['r']
        all_rows.append({
            'category': f'Tau-Amp Corr', 'metric': f'{mode} ρ',
            'mean': np.mean(rs), 'std': np.std(rs),
        })
    # SI (Amp)
    for mode in modes:
        vals = [e for m in all_metrics for e in [
            next(x for x in m['si_amp'] if x['mode'] == mode)
        ]]
        si_vals = [v['mean_si'] for v in vals]
        all_rows.append({
            'category': 'SI (Amp)', 'metric': f'{mode} mean SI',
            'mean': np.mean(si_vals), 'std': np.std(si_vals),
        })
    # sync deg stats
    for mode in modes:
        for k in ['mean_deg', 'gini']:
            vals = sd_agg[mode][k]
            all_rows.append({
                'category': f'Sync Deg ({mode})', 'metric': k,
                'mean': np.mean(vals), 'std': np.std(vals),
            })
    # sync-tau corr
    for mode in modes:
        rs = st_agg[mode]['r']
        all_rows.append({
            'category': 'Sync-Tau Corr', 'metric': f'{mode} ρ',
            'mean': np.mean(rs), 'std': np.std(rs),
        })

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(out_dir, 'all_metrics_summary.csv'), index=False)

    # ============================
    # (9) Single-panel dashboard combining all metrics
    # ============================
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    # (a) Amp stats — skewness
    ax_a = fig.add_subplot(gs[0, 0])
    k = 'skewness'
    means = [np.mean(amp_agg[m][k]) for m in modes]
    stds_ = [np.std(amp_agg[m][k]) for m in modes]
    ax_a.bar(modes, means, yerr=stds_, color=[COLOR[m] for m in modes],
             capsize=4, alpha=0.8, edgecolor='black')
    for i, mode in enumerate(modes):
        ax_a.scatter([i] * len(amp_agg[mode][k]), amp_agg[mode][k],
                     color='black', zorder=5, s=15, alpha=0.5)
    ax_a.set_title('(a) Skewness')
    ax_a.set_ylabel('Skewness')

    # (b) Amp stats — kurtosis
    ax_b = fig.add_subplot(gs[0, 1])
    k = 'kurtosis'
    means = [np.mean(amp_agg[m][k]) for m in modes]
    stds_ = [np.std(amp_agg[m][k]) for m in modes]
    ax_b.bar(modes, means, yerr=stds_, color=[COLOR[m] for m in modes],
             capsize=4, alpha=0.8, edgecolor='black')
    for i, mode in enumerate(modes):
        ax_b.scatter([i] * len(amp_agg[mode][k]), amp_agg[mode][k],
                     color='black', zorder=5, s=15, alpha=0.5)
    ax_b.set_title('(b) Kurtosis')

    # (c) Amp stats — gini
    ax_c = fig.add_subplot(gs[0, 2])
    k = 'gini'
    means = [np.mean(amp_agg[m][k]) for m in modes]
    stds_ = [np.std(amp_agg[m][k]) for m in modes]
    ax_c.bar(modes, means, yerr=stds_, color=[COLOR[m] for m in modes],
             capsize=4, alpha=0.8, edgecolor='black')
    for i, mode in enumerate(modes):
        ax_c.scatter([i] * len(amp_agg[mode][k]), amp_agg[mode][k],
                     color='black', zorder=5, s=15, alpha=0.5)
    ax_c.set_title('(c) Gini Coefficient')

    # (d) Tau–Amp Correlation
    ax_d = fig.add_subplot(gs[1, 0])
    means = [np.mean(tau_corr_agg[m]['r']) for m in modes]
    stds_ = [np.std(tau_corr_agg[m]['r']) for m in modes]
    ax_d.bar(modes, means, yerr=stds_, color=[COLOR[m] for m in modes],
             capsize=4, alpha=0.8, edgecolor='black')
    for i, mode in enumerate(modes):
        ax_d.scatter([i] * len(tau_corr_agg[mode]['r']), tau_corr_agg[mode]['r'],
                     color='black', zorder=5, s=15, alpha=0.5)
    ax_d.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_d.set_title('(d) Tau–Amp Corr (ρ)')
    ax_d.set_ylabel('Spearman ρ')

    # (e) Dominant Amp Corr Matrix (mean)
    ax_e = fig.add_subplot(gs[1, 1])
    im_e = ax_e.imshow(mean_corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax_e.set_xticks(range(4)); ax_e.set_yticks(range(4))
    ax_e.set_xticklabels(modes, fontsize=10); ax_e.set_yticklabels(modes, fontsize=10)
    for i in range(4):
        for j in range(4):
            ax_e.text(j, i, f'{mean_corr[i, j]:.2f}', ha='center', va='center', fontsize=10)
    ax_e.set_title('(e) Dom. Amp Corr (mean)')
    plt.colorbar(im_e, ax=ax_e, shrink=0.8)

    # (f) SI Amplitude
    ax_f = fig.add_subplot(gs[1, 2])
    si_agg_amp = {mode: [] for mode in modes}
    for m in all_metrics:
        for entry in m['si_amp']:
            if entry['mode'] in si_agg_amp:
                si_agg_amp[entry['mode']].append(entry['mean_si'])
    means = [np.mean(si_agg_amp[m]) for m in modes]
    stds_ = [np.std(si_agg_amp[m]) for m in modes]
    ax_f.bar(modes, means, yerr=stds_, color=[COLOR[m] for m in modes],
             capsize=4, alpha=0.8, edgecolor='black')
    for i, mode in enumerate(modes):
        ax_f.scatter([i] * len(si_agg_amp[mode]), si_agg_amp[mode],
                     color='black', zorder=5, s=15, alpha=0.5)
    ax_f.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_f.set_title('(f) Selectivity Index (Amp)')
    ax_f.set_ylabel('Mean SI')

    # (g) Sync Degree–Tau Corr
    ax_g = fig.add_subplot(gs[2, 0])
    means = [np.mean(st_agg[m]['r']) for m in modes]
    stds_ = [np.std(st_agg[m]['r']) for m in modes]
    ax_g.bar(modes, means, yerr=stds_, color=[COLOR[m] for m in modes],
             capsize=4, alpha=0.8, edgecolor='black')
    for i, mode in enumerate(modes):
        ax_g.scatter([i] * len(st_agg[mode]['r']), st_agg[mode]['r'],
                     color='black', zorder=5, s=15, alpha=0.5)
    ax_g.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_g.set_title('(g) Sync Deg–Tau Corr (ρ)')
    ax_g.set_ylabel('Spearman ρ')

    # (h) Jaccard Matrix (mean)
    ax_h = fig.add_subplot(gs[2, 1])
    im_h = ax_h.imshow(mean_jac, vmin=0, vmax=1, cmap='YlOrRd')
    ax_h.set_xticks(range(4)); ax_h.set_yticks(range(4))
    ax_h.set_xticklabels(modes, fontsize=10); ax_h.set_yticklabels(modes, fontsize=10)
    for i in range(4):
        for j in range(4):
            ax_h.text(j, i, f'{mean_jac[i, j]:.2f}', ha='center', va='center', fontsize=10)
    ax_h.set_title(f'(h) Active Group Jaccard (Top 20%)')
    plt.colorbar(im_h, ax=ax_h, shrink=0.8)

    # (i) Sync Degree Gini
    ax_i = fig.add_subplot(gs[2, 2])
    k = 'gini'
    means = [np.mean(sd_agg[m][k]) for m in modes]
    stds_ = [np.std(sd_agg[m][k]) for m in modes]
    ax_i.bar(modes, means, yerr=stds_, color=[COLOR[m] for m in modes],
             capsize=4, alpha=0.8, edgecolor='black')
    for i, mode in enumerate(modes):
        ax_i.scatter([i] * len(sd_agg[mode][k]), sd_agg[mode][k],
                     color='black', zorder=5, s=15, alpha=0.5)
    ax_i.set_title('(i) Sync Degree Gini')
    ax_i.set_ylabel('Gini')

    fig.suptitle(f'Summary Dashboard (n={n_runs} runs)', fontsize=18, y=0.98)
    fig.savefig(os.path.join(out_dir, 'dashboard.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'dashboard.pdf'), bbox_inches='tight')
    plt.close(fig)

    print(f"\n=== All outputs saved to {out_dir}/ ===")
    print("CSV files:")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.csv'):
            print(f"  {f}")
    print("Figures:")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.png'):
            print(f"  {f}")


# ---------- JSON loader ----------

def load_run_from_json(run_path):
    """
    Read ``analysis_results.json`` and convert it into the metrics dict
    format expected by ``aggregate_and_plot``.

    Falls back to ``load_run`` (model-based recomputation) if the JSON is
    not present.
    """
    json_path = os.path.join(run_path, 'figures', 'single', 'analysis_results.json')

    if not os.path.exists(json_path):
        print(f"  WARNING: {json_path} not found. Falling back to model-based computation...")
        return load_run(run_path), False

    with open(json_path, 'r') as f:
        ar = json.load(f)

    print(f"  Loaded from {json_path}")

    metrics = {}

    # --- (0) min_loss ---
    metrics['min_loss'] = ar['min_loss']

    # --- (1) amp_stats (same structure) ---
    metrics['amp_stats'] = ar['amp_stats']

    # --- (2) tau_amp_corr (key rename: tau_amp_correlation → tau_amp_corr) ---
    metrics['tau_amp_corr'] = [
        {'mode': e['mode'], 'r_spearman': e['r_spearman'], 'p_spearman': e['p_spearman']}
        for e in ar['tau_amp_correlation']
    ]

    # --- (3) dominant_amp_corr_matrix (dict-of-dicts → numpy 4×4) ---
    labels = ar['labels']
    corr_dict = ar['dominant_amp_correlation_matrix']
    n = len(labels)
    corr_matrix = np.zeros((n, n))
    for i, m1 in enumerate(labels):
        for j, m2 in enumerate(labels):
            corr_matrix[i, j] = corr_dict[m1][m2]
    metrics['dominant_amp_corr_matrix'] = corr_matrix
    metrics['dominant_amp_corr_labels'] = list(labels)

    # --- (4) si_power (key rename: si_power_stats → si_power) ---
    metrics['si_power'] = ar['si_power_stats']

    # --- (5) si_amp (key rename: si_amp_stats → si_amp) ---
    metrics['si_amp'] = ar['si_amp_stats']

    # --- (6) sync_degree_stats (direct) ---
    metrics['sync_degree_stats'] = ar['sync_degree_stats']

    # --- (6b) sync_degree_corr_matrix (dict-of-dicts → numpy 4×4) ---
    if 'sync_degree_corr_matrix' in ar:
        sd_corr_dict = ar['sync_degree_corr_matrix']
        n = len(labels)
        sd_corr_matrix = np.zeros((n, n))
        for i, m1 in enumerate(labels):
            for j, m2 in enumerate(labels):
                sd_corr_matrix[i, j] = sd_corr_dict[m1][m2]
        metrics['sync_degree_corr_matrix'] = sd_corr_matrix
        metrics['sync_degree_corr_labels'] = list(labels)

    # --- (7) sync_tau_corr (direct) ---
    metrics['sync_tau_corr'] = ar['sync_tau_corr']

    # --- (8) jaccard_matrix (list-of-lists → numpy array) ---
    metrics['jaccard_matrix'] = np.array(ar['jaccard_matrix'])
    metrics['jaccard_labels'] = list(labels)

    return metrics, True


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description='Aggregate analysis across multiple runs')
    parser.add_argument('--runs_dir', type=str, default='multiple_runs',
                        help='Directory containing run subfolders (default: multiple_runs)')
    parser.add_argument('--num_runs', type=int, default=None,
                        help='Number of runs to process (default: auto-detect)')
    parser.add_argument('--start_run', type=int, default=1,
                        help='Starting run number (default: 1)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: <runs_dir>/summary)')
    parser.add_argument('--force_recompute', action='store_true',
                        help='Force model-based recomputation even if JSON exists')
    args = parser.parse_args()

    runs_dir = args.runs_dir
    out_dir = args.out_dir or os.path.join(runs_dir, 'summary')

    # Limit GPU memory (required when falling back to model-based computation)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Detect available runs
    if args.num_runs is not None:
        run_ids = list(range(args.start_run, args.start_run + args.num_runs))
    else:
        run_ids = sorted(
            int(d) for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d)) and d.isdigit()
        )

    print(f"Detected runs: {run_ids}")
    print(f"Output directory: {out_dir}")

    all_metrics = []
    json_count = 0
    fallback_count = 0

    for rid in run_ids:
        run_path = os.path.join(runs_dir, str(rid))
        print(f"\n--- Loading Run {rid} ({run_path}) ---")
        try:
            if args.force_recompute:
                metrics = load_run(run_path)
                fallback_count += 1
            else:
                metrics, from_json = load_run_from_json(run_path)
                if from_json:
                    json_count += 1
                else:
                    fallback_count += 1
            all_metrics.append(metrics)
            print(f"  min_loss = {metrics['min_loss']:.6e}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if len(all_metrics) < 2:
        print("ERROR: Need at least 2 successful runs for aggregation.")
        sys.exit(1)

    print(f"\nSuccessfully loaded {len(all_metrics)} / {len(run_ids)} runs.")
    print(f"  From JSON: {json_count}, Fallback (model): {fallback_count}")
    aggregate_and_plot(all_metrics, out_dir)


if __name__ == '__main__':
    main()
