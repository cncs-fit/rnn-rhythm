"""
Figure 6: visualization of the phase-interference mechanism (representative Run 2).
- Top row: alpha -> beta transition of the population output z(t) (time series + PSD).
- Bottom row: same transition for a representative neuron (time series + PSD).
=> z(t) changes drastically while the single-neuron oscillation does not.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal as sp_signal
from function import *

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
})

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

color = {'theta': 'orange', 'alpha': 'indianred', 'beta': 'mediumaquamarine', 'gamma': 'mediumpurple'}

# ===== Load the trained model (Run 2) =====
run_id = 2
run_path = f"multiple_runs/{run_id}/results"
model, dsg, history = load_from_json(run_path)
model.load_weights(f"{run_path}/checkpoints/best_model.weights.h5")
try:
    model.sort_by_tau()
except:
    pass

alpha_param = model.rnn_layer.cell.alpha.numpy()
tau = 1.0 / np.array(alpha_param).flatten()

# ===== Generate evaluation data =====
dsg.update_task_config(period_length=8000, switch_num=dsg.task.n_rhythm)
dsg.update_noise_config(strength=0)

inputs, noise, mask, onehots, labels = dsg.make_datasets(
    batch_size=1, specified_rhythm=dsg.task.rhythm_names)
labels = labels[0]
init_state = model.get_initial_state(batch_size=1)

y, z = model(inputs, noise, init_state)
z_np = z[0, :, 0].numpy() if hasattr(z[0, :, 0], 'numpy') else np.array(z[0, :, 0])
r = model.r[0].numpy().T  # (units, time)

units = model.rnn_layer.cell.units
Fs = dsg.Fs
wait_length = dsg.task.wait_length
period_length = dsg.task.period_length
switch_num = dsg.task.switch_num
rhythms = dsg.task.rhythms

alpha_idx = list(labels).index('alpha')
beta_idx = list(labels).index('beta')
alpha_low, alpha_high = rhythms['alpha']
beta_low, beta_high = rhythms['beta']

print(f"Run {run_id}: labels={labels}")
print(f"  wait={wait_length}, period={period_length}, Fs={Fs}")

# ===== Slice the continuous time-series segments =====
# alpha interval starts at: wait_length + alpha_idx * period_length
# beta  interval starts at: wait_length + beta_idx  * period_length
t_alpha_start = wait_length + alpha_idx * period_length
t_alpha_end = t_alpha_start + period_length
t_beta_start = wait_length + beta_idx * period_length
t_beta_end = t_beta_start + period_length

# alpha -> beta must be adjacent intervals (alpha_idx=1, beta_idx=2)
assert beta_idx == alpha_idx + 1, "alpha and beta should be adjacent"
t_start = t_alpha_start
t_end = t_beta_end

# Visualization window centered on the transition
margin = 1000  # show 1000 ms on each side of the transition point
vis_start = max(t_alpha_start + 2000, 0)  # start from the middle of the alpha interval
vis_end = min(t_beta_start + 5000, t_beta_end)  # up to the middle of the beta interval
transition_t = t_beta_start  # transition point

# Time axis (ms)
t_vis = np.arange(vis_start, vis_end) / Fs * 1000
t_transition_ms = transition_t / Fs * 1000

# Continuous z signal
z_vis = z_np[vis_start:vis_end]

# Pick the representative neuron: the one with the largest alpha amplitude
r_split = split_signal(r, wait_length, switch_num)
filtered_alpha_split = bandpass_filter(r_split, alpha_low, alpha_high, Fs)
_, env_alpha_split, _ = hilbert(filtered_alpha_split)
amp_alpha_mode = np.mean(env_alpha_split[alpha_idx, :, 500:], axis=-1)
nid = np.argmax(amp_alpha_mode)
print(f"  Representative neuron: {nid} (τ={tau[nid]:.0f})")

# Continuous signal of the chosen neuron
r_neuron_vis = r[nid, vis_start:vis_end]

# Apply bandpass filters to the continuous signal
def bp(sig, low, high, fs):
    nyq = 0.5 * fs
    b, a = sp_signal.butter(4, [low/nyq, high/nyq], btype='band')
    return sp_signal.filtfilt(b, a, sig)

z_filt_alpha = bp(z_vis, alpha_low, alpha_high, Fs)
z_filt_beta = bp(z_vis, beta_low, beta_high, Fs)
r_filt_alpha = bp(r_neuron_vis, alpha_low, alpha_high, Fs)
r_filt_beta = bp(r_neuron_vis, beta_low, beta_high, Fs)

# Steady-state segments used for PSD estimation (sufficiently far from the transition)
psd_cut = 500  # drop the leading transient
z_alpha_seg = z_np[t_alpha_start + psd_cut : t_alpha_end]
z_beta_seg = z_np[t_beta_start + psd_cut : t_beta_end]
r_alpha_seg = r[nid, t_alpha_start + psd_cut : t_alpha_end]
r_beta_seg = r[nid, t_beta_start + psd_cut : t_beta_end]

# ===== Compute kappa and eta (reported in the main text) =====
_, env_alpha_all, pha_alpha_all = hilbert(filtered_alpha_split)
amp_in_alpha = np.mean(env_alpha_all[alpha_idx, :, psd_cut:], axis=-1)
amp_in_beta = np.mean(env_alpha_all[beta_idx, :, psd_cut:], axis=-1)
kappa = amp_in_beta / (amp_in_alpha + 1e-20)

# R(t) and η
def compute_eta(env, pha, cut_s):
    e = env[:, cut_s:]
    p = pha[:, cut_s:]
    R_t = np.abs(np.mean(e * np.exp(1j * p), axis=0))
    scalar_t = np.mean(e, axis=0)
    return np.mean(R_t) / (np.mean(scalar_t) + 1e-20)

eta_alpha_alpha = compute_eta(env_alpha_all[alpha_idx], pha_alpha_all[alpha_idx], psd_cut)
eta_beta_alpha = compute_eta(env_alpha_all[beta_idx], pha_alpha_all[beta_idx], psd_cut)

print(f"\n--- Quantitative summary (for paper text) ---")
print(f"  mean κ = {np.mean(kappa):.2f} ± {np.std(kappa):.2f}")
print(f"  median κ = {np.median(kappa):.2f}")
print(f"  κ > 0.5: {np.sum(kappa > 0.5)}/{units}")
print(f"  κ > 1.0: {np.sum(kappa > 1.0)}/{units}")
print(f"  η^(α,α) = {eta_alpha_alpha:.4f}")
print(f"  η^(β,α) = {eta_beta_alpha:.6f}")

# Alpha-band power of the representative neuron
f_a, p_a = sp_signal.welch(r_alpha_seg, fs=Fs, nperseg=1024)
f_b, p_b = sp_signal.welch(r_beta_seg, fs=Fs, nperseg=1024)
df = f_a[1] - f_a[0]
mask_ab = (f_a >= alpha_low) & (f_a <= alpha_high)
pow_neuron_alpha_in_alpha = np.sum(p_a[mask_ab]) * df
pow_neuron_alpha_in_beta = np.sum(p_b[mask_ab]) * df
print(f"  Neuron {nid} α-band power: α-mode={pow_neuron_alpha_in_alpha:.4f}, β-mode={pow_neuron_alpha_in_beta:.4f} (ratio={pow_neuron_alpha_in_beta/pow_neuron_alpha_in_alpha:.2f})")

# ===== Figure 6 =====
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig, width_ratios=[3, 1, 1],
              hspace=0.35, wspace=0.35)

# Optional truncation of the display range (for readability)
show_s = 0
show_e = len(t_vis)
t_show = t_vis[show_s:show_e]

# --- (A) Continuous z(t) time series ---
ax_z = fig.add_subplot(gs[0, 0])
ax_z.plot(t_show, z_vis[show_s:show_e], color='k', linewidth=0.5, alpha=0.4, zorder=1)
ax_z.plot(t_show, z_filt_beta[show_s:show_e], color=color['beta'], linewidth=1.0, label=r'$\beta$ component', zorder=2)
ax_z.plot(t_show, z_filt_alpha[show_s:show_e], color=color['alpha'], linewidth=1.0, label=r'$\alpha$ component', zorder=3)
ax_z.axvline(t_transition_ms, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
# Mode labels
y_top = ax_z.get_ylim()[1] * 0.85 if ax_z.get_ylim()[1] > 0 else 0.01
ax_z.text(t_transition_ms - 500, ax_z.get_ylim()[1] * 0.9, r'$\alpha$ mode',
          ha='right', va='top', fontsize=12, color=color['alpha'], fontweight='bold')
ax_z.text(t_transition_ms + 500, ax_z.get_ylim()[1] * 0.9, r'$\beta$ mode',
          ha='left', va='top', fontsize=12, color=color['beta'], fontweight='bold')
ax_z.set_ylabel("Collective output $z(t)$")
ax_z.set_title("(A) Collective output $z(t)$", fontsize=14)
ax_z.legend(loc='lower right', fontsize=9)
ax_z.set_xlabel("Time (ms)")

# --- (B) Continuous single-neuron time series ---
ax_n = fig.add_subplot(gs[1, 0])
ax_n.plot(t_show, r_neuron_vis[show_s:show_e], color='k', linewidth=0.5, alpha=0.4)
ax_n.plot(t_show, r_filt_alpha[show_s:show_e], color=color['alpha'], linewidth=1.0, label=r'$\alpha$ component')
ax_n.plot(t_show, r_filt_beta[show_s:show_e], color=color['beta'], linewidth=1.0, label=r'$\beta$ component')
ax_n.axvline(t_transition_ms, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax_n.set_ylabel(f"Neuron {nid} ($\\tau$={tau[nid]:.0f})")
ax_n.set_title(f"(B) Representative neuron (neuron {nid}, $\\tau$={tau[nid]:.0f})", fontsize=14)
ax_n.legend(loc='lower right', fontsize=9)
ax_n.set_xlabel("Time (ms)")

# --- (C) PSD of z: alpha interval ---
ax_psd_z_a = fig.add_subplot(gs[0, 1])
freqs, psd = sp_signal.welch(z_alpha_seg, fs=Fs, nperseg=min(2048, len(z_alpha_seg)))
ax_psd_z_a.semilogy(freqs, psd, 'k-', linewidth=1)
ax_psd_z_a.axvspan(alpha_low, alpha_high, alpha=0.25, color=color['alpha'])
ax_psd_z_a.axvspan(beta_low, beta_high, alpha=0.25, color=color['beta'])
ax_psd_z_a.set_xlim(0, 55)
ax_psd_z_a.set_xlabel("Freq (Hz)")
ax_psd_z_a.set_ylabel("PSD")
ax_psd_z_a.set_title(r"(C) $z$: $\alpha$ mode", fontsize=13)

# --- (D) PSD of z: beta interval ---
ax_psd_z_b = fig.add_subplot(gs[0, 2])
freqs, psd = sp_signal.welch(z_beta_seg, fs=Fs, nperseg=min(2048, len(z_beta_seg)))
ax_psd_z_b.semilogy(freqs, psd, 'k-', linewidth=1)
ax_psd_z_b.axvspan(alpha_low, alpha_high, alpha=0.25, color=color['alpha'])
ax_psd_z_b.axvspan(beta_low, beta_high, alpha=0.25, color=color['beta'])
ax_psd_z_b.set_xlim(0, 55)
ax_psd_z_b.set_xlabel("Freq (Hz)")
ax_psd_z_b.set_title(r"(D) $z$: $\beta$ mode", fontsize=13)

# --- (E) PSD of the neuron: alpha interval ---
ax_psd_n_a = fig.add_subplot(gs[1, 1])
freqs, psd = sp_signal.welch(r_alpha_seg, fs=Fs, nperseg=1024)
ax_psd_n_a.semilogy(freqs, psd, 'k-', linewidth=1)
ax_psd_n_a.axvspan(alpha_low, alpha_high, alpha=0.25, color=color['alpha'])
ax_psd_n_a.axvspan(beta_low, beta_high, alpha=0.25, color=color['beta'])
ax_psd_n_a.set_xlim(0, 55)
ax_psd_n_a.set_xlabel("Freq (Hz)")
ax_psd_n_a.set_ylabel("PSD")
ax_psd_n_a.set_title(r"(E) Neuron: $\alpha$ mode", fontsize=13)

# --- (F) PSD of the neuron: beta interval ---
ax_psd_n_b = fig.add_subplot(gs[1, 2])
freqs, psd = sp_signal.welch(r_beta_seg, fs=Fs, nperseg=1024)
ax_psd_n_b.semilogy(freqs, psd, 'k-', linewidth=1)
ax_psd_n_b.axvspan(alpha_low, alpha_high, alpha=0.25, color=color['alpha'])
ax_psd_n_b.axvspan(beta_low, beta_high, alpha=0.25, color=color['beta'])
ax_psd_n_b.set_xlim(0, 55)
ax_psd_n_b.set_xlabel("Freq (Hz)")
ax_psd_n_b.set_title(r"(F) Neuron: $\beta$ mode", fontsize=13)

fig.suptitle(f"Run {run_id}: Phase interference mechanism ($\\alpha \\to \\beta$ transition)",
             fontsize=15, y=0.99)

fig_path = f"multiple_runs/{run_id}/figures/phase_coherence"
os.makedirs(fig_path, exist_ok=True)
plt.savefig(f"{fig_path}/fig6_phase_interference.png", dpi=200, bbox_inches='tight')
plt.savefig(f"{fig_path}/fig6_phase_interference.pdf", bbox_inches='tight')
print(f"\nFigure saved to {fig_path}/fig6_phase_interference.pdf")

