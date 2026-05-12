"""
Phase 2: Fixed-point analysis and free-run simulation for baseline shift mechanism.

For selected runs (high beta-gamma overlap), this script:
1. Loads model and generates data
2. Finds true fixed points via optimization (Sussillo & Barak method)  
3. Computes discrete-time Jacobian eigenvalues at each fixed point
4. Runs free-run simulation from each mode's baseline
5. Produces Figure 5 panels
"""
import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import signal as sig

from models.leaky_rnn_model import ATCLRNNModel
from dataset_generator import DatasetGenerator
from function import load_from_json, split_signal

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

FS = 1000  # sampling frequency (Hz)
RHYTHM_NAMES = ['theta', 'alpha', 'beta', 'gamma']

# ============================================================
# Data loading (adapted from analysis_dynamics.py)
# ============================================================

def get_model_and_data(path, period_length=8000):
    """Load model and generate data with specified rhythm sequence."""
    model, dsg, history = load_from_json(path)

    weights_path = os.path.join(path, "checkpoints", "best_model.weights.h5")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(path, "checkpoints", "last_weight.weights.h5")
    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path)

    # Sequential presentation: theta, alpha, beta, gamma
    dsg.update_task_config(period_length=period_length, switch_num=dsg.task.n_rhythm)
    dsg.update_noise_config(strength=0)

    inputs, noise, mask, onehots, labels = dsg.make_datasets(
        batch_size=1, specified_rhythm=dsg.task.rhythm_names
    )
    init_state = model.get_initial_state(batch_size=1)
    y, z_avg = model(inputs, noise, init_state)

    # r is the firing rate (after tanh): shape (time, units)
    r = model.r[0].numpy()
    return model, dsg, r, labels[0]


def extract_weights(model):
    """Extract alpha, W_in, W_rec, bias from model."""
    weights = model.rnn_layer.get_weights()
    alpha = np.array(weights[0]).flatten()
    W_in = np.array(weights[1])
    W_rec = np.array(weights[2])
    bias = np.array(weights[3])
    return alpha, W_in, W_rec, bias


def split_by_mode(r, dsg):
    """Split firing rates by mode. Returns dict {mode_name: (time, units)}."""
    wait_length = dsg.task.wait_length
    switch_num = dsg.task.switch_num
    r_T = r.T  # (units, time)
    r_split_T = split_signal(r_T, wait_length, switch_num)  # (switch_num, units, period_len)
    r_split = np.transpose(r_split_T, (0, 2, 1))  # (switch_num, period_len, units)
    result = {}
    for i, name in enumerate(RHYTHM_NAMES):
        result[name] = r_split[i]
    return result


# ============================================================
# Fixed-point finding (Sussillo & Barak method)
# ============================================================

def dynamics_map(x, alpha, W_rec, bias):
    """
    One step of the discrete-time dynamics (no input, no noise):
      x_{t+1} = (1-alpha)*x_t + alpha*(tanh(x_t) @ W_rec + bias)
    
    Note: In TF/Keras convention, the recurrent term is r @ W_rec
    where r = tanh(x), and W_rec is (units, units) with [from, to].
    """
    r = np.tanh(x)
    return (1 - alpha) * x + alpha * (r @ W_rec + bias)


def fixed_point_loss(x, alpha, W_rec, bias):
    """||f(x) - x||^2, the quantity to minimize for fixed-point search."""
    fx = dynamics_map(x, alpha, W_rec, bias)
    diff = fx - x
    return np.sum(diff ** 2)


def fixed_point_grad(x, alpha, W_rec, bias):
    """Gradient of the fixed-point loss q(x) = ||f(x)-x||^2 with respect to x."""
    r = np.tanh(x)
    fx = (1 - alpha) * x + alpha * (r @ W_rec + bias)
    diff = fx - x  # f(x) - x
    
    # d(f(x))/dx: Jacobian of the map
    # J_ij = (1-alpha_i)*delta_ij + alpha_i * (1-r_j^2) * W_rec_ji
    # But we need df_i/dx_j:
    # f_i = (1-alpha_i)*x_i + alpha_i * (sum_k tanh(x_k) * W_rec_{k,i} + bias_i)
    # df_i/dx_j = (1-alpha_i)*delta_ij + alpha_i * (1-tanh(x_j)^2) * W_rec_{j,i}
    # So: dq/dx_j = 2 * sum_i diff_i * (df_i/dx_j - delta_ij)
    #            = 2 * sum_i diff_i * (-alpha_i*delta_ij + alpha_i*(1-r_j^2)*W_rec_{j,i})
    
    sech2 = 1 - r ** 2  # (units,)
    
    # More efficient: compute J^T @ diff - diff
    # (J - I)^T @ diff where J is the discrete Jacobian
    # (J-I)_ij = -alpha_i*delta_ij + alpha_i*(1-r_j^2)*W_rec_{j,i}
    # ((J-I)^T)_ji = (J-I)_ij
    # So ((J-I)^T @ diff)_j = sum_i diff_i * (-alpha_i*delta_ij + alpha_i*(1-r_j^2)*W_rec_{j,i})
    #                        = -alpha_j*diff_j + (1-r_j^2) * sum_i alpha_i*diff_i*W_rec_{j,i}
    
    alpha_diff = alpha * diff  # (units,)
    grad = -alpha * diff + sech2 * (alpha_diff @ W_rec.T)
    # This is ((J-I)^T @ diff) -- but W_rec in Keras is [from, to], 
    # so W_rec_{j,i} means row j, col i => W_rec[j,i]
    # sum_i alpha_diff_i * W_rec_{j,i} = (alpha_diff) @ W_rec^T at position j? 
    # No: sum_i v_i * W_rec_{j,i} = sum_i v_i * W_rec[j,i] = (W_rec @ v)[j] 
    # Wait, (A @ v)_j = sum_i A[j,i]*v[i]. But W_rec is (units, units) with [from, to].
    # W_rec[j,i] = weight from j to i. So sum_i alpha_diff_i * W_rec[j,i] = (alpha_diff @ W_rec^T... 
    # No: we need sum_i alpha_diff[i] * W_rec[j,i].
    # W_rec[j,i] is entry (j,i). sum_i x[i]*A[j,i] = this is just A[j,:] @ x = (A @ x)[j].
    # So: sum_i alpha_diff[i] * W_rec[j,i] = (W_rec @ alpha_diff)[j]
    
    # Let me redo: W_rec has shape (units, units). In Keras, prev_output @ W_rec means
    # output_i = sum_j prev_j * W_rec[j,i]. So W_rec[j,i] = weight from unit j to unit i.
    # 
    # f_i = (1-alpha_i)*x_i + alpha_i * (sum_k r_k * W_rec[k,i] + bias_i)
    # df_i/dx_j = (1-alpha_i)*delta_ij + alpha_i * sech2_j * W_rec[j,i]
    # 
    # grad_j = 2 * sum_i diff_i * (df_i/dx_j - delta_ij)
    #        = 2 * sum_i diff_i * (-alpha_i*delta_ij + alpha_i*sech2_j*W_rec[j,i])
    #        = 2 * (-alpha_j*diff_j + sech2_j * sum_i (alpha_i*diff_i) * W_rec[j,i])
    #        = 2 * (-alpha_j*diff_j + sech2_j * (W_rec @ alpha_diff)[j])
    # because sum_i v[i]*W_rec[j,i] = W_rec[j,:] @ v = (W_rec @ v)[j]... 
    # No! W_rec[j,i] -> row j, col i. W_rec[j,:] is row j. (W_rec[j,:] @ v) = sum_i W_rec[j,i]*v[i]. Yes!
    # So (W_rec @ v)[j] = sum_i W_rec[j,i]*v[i]. Correct.
    
    grad = 2.0 * (-alpha * diff + sech2 * (W_rec @ alpha_diff))
    return grad


def find_fixed_points(alpha, W_rec, bias, initial_guesses, tol=1e-10, verbose=True):
    """
    Find fixed points starting from multiple initial guesses.
    Returns list of (x_star, q_value) for converged points.
    """
    found = []
    for i, x0 in enumerate(initial_guesses):
        result = minimize(
            fixed_point_loss,
            x0,
            args=(alpha, W_rec, bias),
            jac=fixed_point_grad,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-12}
        )
        q = result.fun
        if verbose:
            print(f"  Init {i}: q = {q:.2e}, converged={result.success}, nit={result.nit}")
        found.append((result.x, q))
    return found


def cluster_fixed_points(fps, dist_threshold=1.0):
    """Cluster nearby fixed points. Returns representative of each cluster."""
    if not fps:
        return []
    clusters = []
    for x, q in fps:
        merged = False
        for cluster in clusters:
            if np.linalg.norm(x - cluster['x']) < dist_threshold:
                if q < cluster['q']:
                    cluster['x'] = x
                    cluster['q'] = q
                cluster['count'] += 1
                merged = True
                break
        if not merged:
            clusters.append({'x': x, 'q': q, 'count': 1})
    return clusters


# ============================================================
# Jacobian and eigenvalue analysis
# ============================================================

def discrete_jacobian(x_star, alpha, W_rec):
    """
    Discrete-time Jacobian of the map f(x) at x*:
      J_ij = (1-alpha_i)*delta_ij + alpha_i * sech^2(x*_j) * W_rec[j,i]
    
    In matrix form:
      J = diag(1-alpha) + diag(alpha) @ (diag(sech^2(x*)) @ W_rec)^T ... no.
    
    Actually: J_ij = df_i/dx_j.
    f_i = (1-alpha_i)*x_i + alpha_i*(sum_k tanh(x_k)*W_rec[k,i] + bias_i)
    df_i/dx_j = (1-alpha_i)*delta_ij + alpha_i * sech2(x_j) * W_rec[j,i]
    
    So J[i,j] = (1-alpha[i]) if i==j, plus alpha[i]*sech2[j]*W_rec[j,i]
    
    In matrix form: J = diag(1-alpha) + diag(alpha) @ W_rec.T @ diag(sech2)
    """
    N = len(x_star)
    sech2 = 1 - np.tanh(x_star) ** 2
    J = np.diag(1 - alpha) + np.diag(alpha) @ W_rec.T @ np.diag(sech2)
    return J


def eigenvalue_analysis(J):
    """
    Compute eigenvalues of discrete-time Jacobian.
    Returns eigenvalues, their magnitudes, and oscillation frequencies.
    """
    eigvals = np.linalg.eigvals(J)
    magnitudes = np.abs(eigvals)
    # Frequency from argument: f = |arg(lambda)| / (2*pi) * Fs
    frequencies = np.abs(np.angle(eigvals)) / (2 * np.pi) * FS
    return eigvals, magnitudes, frequencies


# ============================================================
# Free-run simulation
# ============================================================

def free_run(x0, alpha, W_rec, bias, n_steps=5000):
    """
    Simulate autonomous dynamics (no input, no noise) from initial state x0.
    Returns firing rates r(t) of shape (n_steps, units).
    """
    N = len(x0)
    x = x0.copy()
    r_traj = np.zeros((n_steps, N))
    for t in range(n_steps):
        r = np.tanh(x)
        r_traj[t] = r
        x = (1 - alpha) * x + alpha * (r @ W_rec + bias)
    return r_traj


def compute_power_spectrum(z, fs=FS, nperseg=None):
    """Compute power spectrum of 1D signal z using Welch's method."""
    if nperseg is None:
        nperseg = min(len(z), 4096)
    f, psd = sig.welch(z, fs=fs, nperseg=nperseg)
    return f, psd


# ============================================================
# Main analysis
# ============================================================

def analyze_run(run_id, results_base="multiple_runs", period_length=8000, 
                output_dir=None):
    """Full analysis for a single run."""
    exp_path = os.path.join(results_base, str(run_id), "results")
    if output_dir is None:
        output_dir = os.path.join(results_base, str(run_id), "figures", "baseline")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Analyzing Run {run_id}")
    print(f"{'='*60}")

    # --- Load model and data ---
    model, dsg, r, labels = get_model_and_data(exp_path, period_length=period_length)
    alpha, W_in, W_rec, bias = extract_weights(model)
    tau = 1.0 / (alpha + 1e-12)
    r_by_mode = split_by_mode(r, dsg)

    # Skip transient (first 500 steps of each segment)
    transient = 500
    baselines = {}
    for mode_name, r_mode in r_by_mode.items():
        baselines[mode_name] = np.mean(r_mode[transient:], axis=0)

    print(f"\nAlpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
    print(f"Tau range: [{tau.min():.1f}, {tau.max():.1f}] steps")

    # --- Focus on beta and gamma ---
    focus_modes = ['beta', 'gamma']

    # --- Find fixed points ---
    print("\n--- Fixed-point search (beta/gamma) ---")
    all_fps = {}
    for mode_name in focus_modes:
        print(f"\nMode: {mode_name}")
        r_bar = baselines[mode_name]
        # Convert r_bar (firing rates) to x_bar (pre-activation) via arctanh
        x_bar = np.arctanh(np.clip(r_bar, -0.9999, 0.9999))

        # Multiple initial conditions: baseline + perturbations
        n_perturb = 5
        inits = [x_bar.copy()]
        for _ in range(n_perturb):
            inits.append(x_bar + np.random.randn(len(x_bar)) * 0.01)

        fps = find_fixed_points(alpha, W_rec, bias, inits, verbose=True)
        clusters = cluster_fixed_points(fps)
        clusters.sort(key=lambda c: c['q'])

        print(f"  Found {len(clusters)} distinct fixed point(s):")
        for ci, c in enumerate(clusters):
            print(f"    FP{ci}: q={c['q']:.2e}, count={c['count']}")

        all_fps[mode_name] = clusters

    # --- Eigenvalue analysis at best fixed point per mode ---
    print("\n--- Eigenvalue analysis ---")
    eigen_results = {}
    for mode_name in focus_modes:
        if not all_fps[mode_name]:
            print(f"  {mode_name}: No fixed point found!")
            continue
        best_fp = all_fps[mode_name][0]
        x_star = best_fp['x']
        J = discrete_jacobian(x_star, alpha, W_rec)
        eigvals, mags, freqs = eigenvalue_analysis(J)

        # Find dominant unstable eigenvalues (|lambda|>1 and nonzero frequency)
        unstable = mags > 1.0
        oscillatory = freqs > 1.0  # >1 Hz
        dominant_mask = unstable & oscillatory
        
        if np.any(dominant_mask):
            dom_freqs = freqs[dominant_mask]
            dom_mags = mags[dominant_mask]
            # Sort by magnitude (descending)
            sort_idx = np.argsort(dom_mags)[::-1]
            top_freq = dom_freqs[sort_idx[0]]
            top_mag = dom_mags[sort_idx[0]]
            print(f"  {mode_name}: q={best_fp['q']:.2e}, "
                  f"dominant unstable freq={top_freq:.1f} Hz (|λ|={top_mag:.4f}), "
                  f"n_unstable_osc={np.sum(dominant_mask)}")
        else:
            # Check marginally stable
            near_one = np.abs(mags - 1.0) < 0.05
            if np.any(near_one & oscillatory):
                near_freqs = freqs[near_one & oscillatory]
                print(f"  {mode_name}: q={best_fp['q']:.2e}, "
                      f"near-unit-circle freqs={np.sort(near_freqs)[:5]}")
            else:
                top5_idx = np.argsort(mags)[::-1][:5]
                print(f"  {mode_name}: q={best_fp['q']:.2e}, "
                      f"top5 |λ|={mags[top5_idx]}, freqs={freqs[top5_idx]}")

        eigen_results[mode_name] = {
            'eigvals': eigvals,
            'magnitudes': mags,
            'frequencies': freqs,
            'x_star': x_star,
            'q': best_fp['q'],
        }

    # --- Free-run simulation ---
    print("\n--- Free-run simulation ---")
    freerun_results = {}
    for mode_name in focus_modes:
        r_bar = baselines[mode_name]
        x0 = np.arctanh(np.clip(r_bar, -0.9999, 0.9999))
        r_traj = free_run(x0, alpha, W_rec, bias, n_steps=5000)
        z_traj = np.mean(r_traj, axis=1)  # population mean

        # Skip initial transient for spectrum
        z_analysis = z_traj[500:]
        f, psd = compute_power_spectrum(z_analysis)
        peak_freq = f[np.argmax(psd)]
        print(f"  {mode_name}: peak frequency = {peak_freq:.1f} Hz")
        freerun_results[mode_name] = {
            'r_traj': r_traj,
            'z_traj': z_traj,
            'f': f,
            'psd': psd,
            'peak_freq': peak_freq,
        }

    # --- Compute per-neuron amplitudes (variance) for beta/gamma ---
    amplitudes = {}
    for mode_name in focus_modes:
        r_mode = r_by_mode[mode_name][transient:]  # (time, units)
        amplitudes[mode_name] = np.std(r_mode, axis=0)  # per-neuron std

    # --- Generate figures ---
    plot_figure5(eigen_results, freerun_results, baselines, amplitudes,
                 tau, output_dir, run_id)

    # --- Save numerical results ---
    save_results(all_fps, eigen_results, freerun_results, output_dir)

    return all_fps, eigen_results, freerun_results


# ============================================================
# Plotting
# ============================================================

COLORS = {
    'theta': '#E74C3C',
    'alpha': '#27AE60',
    'beta':  '#2980B9',
    'gamma': '#8E44AD',
}

BAND_RANGES = {
    'theta': (4, 7),
    'alpha': (8, 13),
    'beta':  (14, 29),
    'gamma': (30, 50),
}


def plot_figure5(eigen_results, freerun_results, baselines, amplitudes,
                 tau, output_dir, run_id):
    """Generate Figure 5 panels, focused on beta/gamma."""
    focus_modes = ['beta', 'gamma']
    import matplotlib.colors as mcolors

    # Global font size
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })

    fig = plt.figure(figsize=(15, 11))
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4),
    ]

    # Tau colormap (log scale)
    log_tau = np.log10(tau)
    norm = mcolors.Normalize(vmin=log_tau.min(), vmax=log_tau.max())
    cmap = plt.cm.viridis  # type: ignore

    # Precompute baseline shift and amplitude
    r_bar_beta = baselines['beta']
    r_bar_gamma = baselines['gamma']
    baseline_shift = np.abs(r_bar_gamma - r_bar_beta)
    amp_max = np.maximum(amplitudes['beta'], amplitudes['gamma'])
    median_amp = np.median(amp_max)

    # Print quantitative summary for low-amplitude neurons
    low_amp_mask = amp_max <= median_amp
    high_amp_mask = amp_max > median_amp
    print(f"\n--- Baseline shift summary ---")
    print(f"  Median oscillation amplitude: {median_amp:.4f}")
    print(f"  Low-amplitude neurons (n={np.sum(low_amp_mask)}): "
          f"mean baseline shift = {baseline_shift[low_amp_mask].mean():.3f} "
          f"± {baseline_shift[low_amp_mask].std():.3f}")
    print(f"  High-amplitude neurons (n={np.sum(high_amp_mask)}): "
          f"mean baseline shift = {baseline_shift[high_amp_mask].mean():.3f} "
          f"± {baseline_shift[high_amp_mask].std():.3f}")

    # --- Panel A: Baseline scatter (beta vs gamma mean r_i) ---
    ax = axes[0]
    sc = ax.scatter(r_bar_beta, r_bar_gamma, c=log_tau, cmap=cmap, norm=norm,
                    s=35, alpha=0.8, edgecolors='none')
    lims = [min(r_bar_beta.min(), r_bar_gamma.min()) - 0.05,
            max(r_bar_beta.max(), r_bar_gamma.max()) + 0.05]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel(r'Mean activity $\bar{r}_i$ ($\beta$ mode)')
    ax.set_ylabel(r'Mean activity $\bar{r}_i$ ($\gamma$ mode)')
    ax.set_title(r'(A) Baseline: $\beta$ vs $\gamma$')
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(r'$\log_{10}\tau_i$')

    # --- Panel B: Free-run power spectra ---
    ax = axes[1]
    for mode_name in focus_modes:
        fr = freerun_results[mode_name]
        ax.semilogy(fr['f'], fr['psd'], color=COLORS[mode_name],
                    label=fr'$\{mode_name}$ (peak {fr["peak_freq"]:.1f} Hz)',
                    linewidth=2.0)
    for mode_name in focus_modes:
        lo, hi = BAND_RANGES[mode_name]
        ax.axvspan(lo, hi, alpha=0.12, color=COLORS[mode_name])
    ax.set_xlim(0, 55)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power spectral density')
    ax.set_title('(B) Free-run power spectra')
    ax.legend()

    # --- Panel C: Baseline shift vs oscillation amplitude (log x-axis) ---
    ax = axes[2]
    # Shift x slightly to avoid log(0)
    amp_plot = np.clip(amp_max, 1e-4, None)
    sc2 = ax.scatter(amp_plot, baseline_shift, c=log_tau, cmap=cmap, norm=norm,
                     s=35, alpha=0.8, edgecolors='none')
    ax.set_xscale('log')
    ax.axvline(median_amp, color='gray', linestyle=':', linewidth=1.0, alpha=0.6,
               label=f'median amp = {median_amp:.3f}')
    ax.set_xlabel(r'Oscillation amplitude (std of $r_i$)')
    ax.set_ylabel(r'Baseline shift $|\bar{r}_i^{\beta} - \bar{r}_i^{\gamma}|$')
    ax.set_title('(C) Baseline shift vs amplitude')
    cb2 = fig.colorbar(sc2, ax=ax, shrink=0.8)
    cb2.set_label(r'$\log_{10}\tau_i$')
    ax.legend(loc='upper right')

    # --- Panel D: Eigenvalue magnitude vs frequency ---
    ax = axes[3]
    for mode_name in focus_modes:
        if mode_name not in eigen_results:
            continue
        er = eigen_results[mode_name]
        mags = er['magnitudes']
        freqs = er['frequencies']
        mask = (mags > 0.9) & (freqs > 1)
        ax.scatter(freqs[mask], mags[mask],
                   c=COLORS[mode_name], s=30, alpha=0.5,
                   label=fr'$\{mode_name}$')
    for mode_name in focus_modes:
        lo, hi = BAND_RANGES[mode_name]
        ax.axvspan(lo, hi, alpha=0.10, color=COLORS[mode_name])
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'$|\lambda|$')
    ax.set_title('(D) Eigenvalue magnitude vs frequency')
    ax.set_xlim(0, 55)
    ax.legend()

    fig.suptitle(f'Run {run_id}: Baseline shift and fixed-point analysis '
                 r'($\beta$/$\gamma$)', fontsize=17)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig5_baseline_fp.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig5_baseline_fp.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_dir}/fig5_baseline_fp.pdf")


def save_results(all_fps, eigen_results, freerun_results, output_dir):
    """Save numerical results to JSON."""
    out = {}
    focus_modes = ['beta', 'gamma']
    for mode_name in focus_modes:
        mode_data = {}
        # Fixed points
        if all_fps.get(mode_name):
            best = all_fps[mode_name][0]
            mode_data['fp_q'] = float(best['q'])
            mode_data['fp_x_norm'] = float(np.linalg.norm(best['x']))
        # Eigenvalues
        if mode_name in eigen_results:
            er = eigen_results[mode_name]
            mode_data['n_unstable'] = int(np.sum(er['magnitudes'] > 1.0))
            # Top 5 by magnitude
            top5 = np.argsort(er['magnitudes'])[::-1][:5]
            mode_data['top5_magnitudes'] = er['magnitudes'][top5].tolist()
            mode_data['top5_frequencies'] = er['frequencies'][top5].tolist()
        # Free-run
        if mode_name in freerun_results:
            mode_data['freerun_peak_freq'] = float(freerun_results[mode_name]['peak_freq'])
        out[mode_name] = mode_data

    path = os.path.join(output_dir, 'baseline_fp_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {path}")


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Fixed-point and baseline analysis')
    parser.add_argument('--runs', type=int, nargs='+', default=[1, 19],
                        help='Run IDs to analyze')
    parser.add_argument('--period-length', type=int, default=8000,
                        help='Period length for data generation')
    args = parser.parse_args()

    for run_id in args.runs:
        analyze_run(run_id, period_length=args.period_length)


if __name__ == '__main__':
    main()
