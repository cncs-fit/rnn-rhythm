"""
Analysis for Sect. 3.6: phase-interference control of band power.

For each of the 20 trained networks, the following quantities are computed:
1. Coherence ratio eta^(m,k) over all (mode, band) pairs (4 x 4 matrix).
2. Alpha-band amplitude ratio kappa_i = A_i^(alpha, beta mode) / A_i^(alpha, alpha mode).
3. Per-run classification: amplitude-suppression vs phase-interference type.
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy import signal as sp_signal
from function import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# GPU memory configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def compute_coherence_ratio(r_split, rhythms, labels, Fs, cut_s=500):
    """
    Compute the coherence ratio eta^(m,k) over all (mode, band) pairs.
    eta = <R^(k)(t)>_t / <(1/N) sum_i A_i^(k)(t)>_t
    R^(k)(t) = |(1/N) sum_i A_i^(k)(t) exp(i phi_i^(k)(t))|

    Returns: (n_modes, n_bands) array.
    """
    n_modes = len(labels)
    units = r_split.shape[1]
    eta = np.zeros((n_modes, n_modes))  # eta[mode, band]
    
    band_names = list(rhythms.keys())
    
    for k_idx, band_name in enumerate(band_names):
        low, high = rhythms[band_name]
        # Band-pass filter
        filtered = bandpass_filter(r_split, low, high, Fs)  # (n_modes, units, T)
        # Hilbert transform
        _, envelopes, phases = hilbert(filtered)  # (n_modes, units, T)
        
        for m_idx in range(n_modes):
            env = envelopes[m_idx, :, cut_s:]   # (units, T)
            pha = phases[m_idx, :, cut_s:]       # (units, T)
            
            # R^(k)(t) = |(1/N) Σ_i A_i exp(i φ_i)|
            vector_sum_t = np.mean(env * np.exp(1j * pha), axis=0)  # (T,)
            R_t = np.abs(vector_sum_t)  # (T,)
            
            # (1/N) Σ_i A_i(t)
            scalar_mean_t = np.mean(env, axis=0)  # (T,)
            
            # η = <R(t)> / <scalar_mean(t)>
            mean_R = np.mean(R_t)
            mean_scalar = np.mean(scalar_mean_t)
            
            eta[m_idx, k_idx] = mean_R / (mean_scalar + 1e-20)
    
    return eta


def compute_kappa(r_split, rhythms, labels, Fs, cut_s=500):
    """
    Compute the alpha-band amplitude ratio kappa_i = A_i^(alpha, beta_mode) / A_i^(alpha, alpha_mode).

    Returns: dict with kappa array and summary statistics.
    """
    alpha_idx = list(labels).index('alpha')
    beta_idx = list(labels).index('beta')
    
    low, high = rhythms['alpha']
    filtered = bandpass_filter(r_split, low, high, Fs)
    
    # Time-averaged Hilbert envelope
    _, envelopes, _ = hilbert(filtered)
    amp_alpha_mode = np.mean(envelopes[alpha_idx, :, cut_s:], axis=-1)  # (units,)
    amp_beta_mode = np.mean(envelopes[beta_idx, :, cut_s:], axis=-1)    # (units,)
    
    kappa = amp_beta_mode / (amp_alpha_mode + 1e-20)
    
    return {
        'kappa': kappa,
        'mean_kappa': float(np.mean(kappa)),
        'median_kappa': float(np.median(kappa)),
        'n_above_0.5': int(np.sum(kappa > 0.5)),
        'n_above_1.0': int(np.sum(kappa > 1.0)),
        'amp_alpha_mode': amp_alpha_mode,
        'amp_beta_mode': amp_beta_mode,
    }


def compute_cancel_ratio_alpha(r_split, rhythms, labels, Fs, cut_s=500):
    """
    For both the alpha mode and the beta mode, compute the alpha-band cancel_ratio
    cancel_ratio = |sum_i A_i exp(i phi_i)| / sum_i A_i
    (equivalent to computing eta with a different time-averaging convention).
    """
    alpha_idx = list(labels).index('alpha')
    beta_idx = list(labels).index('beta')
    
    low, high = rhythms['alpha']
    filtered = bandpass_filter(r_split, low, high, Fs)
    _, envelopes, phases = hilbert(filtered)
    
    results = {}
    for mode_idx, mode_name in [(alpha_idx, 'alpha'), (beta_idx, 'beta')]:
        env = envelopes[mode_idx, :, cut_s:]
        pha = phases[mode_idx, :, cut_s:]
        
        # Time-averaged amplitude and representative phase
        mean_amp = np.mean(env, axis=-1)
        mean_phase = np.angle(np.mean(np.exp(1j * pha), axis=-1))
        
        vector_sum = np.abs(np.sum(mean_amp * np.exp(1j * mean_phase)))
        scalar_sum = np.sum(mean_amp)
        cancel_ratio = vector_sum / (scalar_sum + 1e-20)
        
        results[mode_name] = {
            'vector_sum': float(vector_sum),
            'scalar_sum': float(scalar_sum),
            'cancel_ratio': float(cancel_ratio),
        }
    
    return results


def analyze_run(run_id):
    """Run the analysis pipeline for a single training run."""
    run_path = f"multiple_runs/{run_id}/results"

    if not os.path.exists(run_path):
        print(f"  Run {run_id}: path not found, skipping")
        return None

    # Load the trained model
    model, dsg, history = load_from_json(run_path)
    checkpoint_path = f"{run_path}/checkpoints/best_model.weights.h5"
    model.load_weights(checkpoint_path)
    
    try:
        model.sort_by_tau()
    except:
        pass
    
    # Generate the evaluation dataset
    dsg.update_task_config(period_length=8000, switch_num=dsg.task.n_rhythm)
    dsg.update_noise_config(strength=0)
    
    inputs, noise, mask, onehots, labels = dsg.make_datasets(
        batch_size=1, specified_rhythm=dsg.task.rhythm_names
    )
    labels = labels[0]
    init_state = model.get_initial_state(batch_size=1)
    
    y, z = model(inputs, noise, init_state)
    r = model.r[0].numpy().T  # (units, time)
    
    Fs = dsg.Fs
    wait_length = dsg.task.wait_length
    switch_num = dsg.task.switch_num
    rhythms = dsg.task.rhythms
    
    # Split into per-interval segments
    r_split = split_signal(r, wait_length, switch_num)

    # 1. Coherence ratio
    eta = compute_coherence_ratio(r_split, rhythms, labels, Fs)

    # 2. kappa (alpha-band amplitude ratio)
    kappa_result = compute_kappa(r_split, rhythms, labels, Fs)
    
    # 3. cancel ratio
    cancel = compute_cancel_ratio_alpha(r_split, rhythms, labels, Fs)
    
    return {
        'run_id': run_id,
        'labels': list(labels),
        'eta': eta,
        'kappa_mean': kappa_result['mean_kappa'],
        'kappa_median': kappa_result['median_kappa'],
        'kappa_n_above_0.5': kappa_result['n_above_0.5'],
        'kappa_n_above_1.0': kappa_result['n_above_1.0'],
        'cancel_alpha_in_alpha': cancel['alpha']['cancel_ratio'],
        'cancel_alpha_in_beta': cancel['beta']['cancel_ratio'],
        'kappa_all': kappa_result['kappa'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=str, default='all',
                        help='Comma-separated run IDs or "all"')
    args = parser.parse_args()
    
    if args.runs == 'all':
        run_ids = list(range(1, 21))
    else:
        run_ids = [int(x) for x in args.runs.split(',')]
    
    all_results = []
    
    for run_id in run_ids:
        print(f"\n{'='*50}")
        print(f"Run {run_id}")
        print(f"{'='*50}")
        result = analyze_run(run_id)
        if result is not None:
            all_results.append(result)
    
    if not all_results:
        print("No results.")
        return
    
    # ===== Summary output =====
    print("\n" + "="*70)
    print("SUMMARY: Phase coherence screening")
    print("="*70)
    
    band_names = ['theta', 'alpha', 'beta', 'gamma']
    
    # --- Statistics of the diagonal of eta (matched mode) ---
    print("\n--- Coherence ratio η^(m,m) (matched mode, diagonal) ---")
    print(f"{'Run':>4s}  {'θ':>6s}  {'α':>6s}  {'β':>6s}  {'γ':>6s}")
    diag_eta = []
    for res in all_results:
        eta = res['eta']
        diag = np.diag(eta)
        diag_eta.append(diag)
        print(f"  {res['run_id']:>2d}  {diag[0]:.4f}  {diag[1]:.4f}  {diag[2]:.4f}  {diag[3]:.4f}")
    diag_eta = np.array(diag_eta)
    print(f"{'mean':>6s}  {diag_eta[:,0].mean():.4f}  {diag_eta[:,1].mean():.4f}  "
          f"{diag_eta[:,2].mean():.4f}  {diag_eta[:,3].mean():.4f}")
    print(f"{'std':>6s}  {diag_eta[:,0].std():.4f}  {diag_eta[:,1].std():.4f}  "
          f"{diag_eta[:,2].std():.4f}  {diag_eta[:,3].std():.4f}")
    
    # --- Statistics of eta^(beta, alpha): alpha-band coherence in the beta mode ---
    print("\n--- η^(β,α): alpha coherence in beta mode ---")
    eta_beta_alpha = []
    for res in all_results:
        beta_idx = res['labels'].index('beta')
        alpha_band_idx = list(band_names).index('alpha')
        val = res['eta'][beta_idx, alpha_band_idx]
        eta_beta_alpha.append(val)
        print(f"  Run {res['run_id']:>2d}: η^(β,α) = {val:.6f}")
    eta_beta_alpha = np.array(eta_beta_alpha)
    print(f"  mean: {eta_beta_alpha.mean():.6f} ± {eta_beta_alpha.std():.6f}")
    
    # --- κ (alpha amplitude ratio) ---
    print("\n--- κ: alpha amplitude ratio (beta mode / alpha mode) ---")
    print(f"{'Run':>4s}  {'mean_κ':>8s}  {'median_κ':>9s}  {'n>0.5':>6s}  {'n>1.0':>6s}  {'Type':>12s}")
    
    for res in all_results:
        # Classification rule: median(kappa) > 0.3 AND at least 20 neurons with kappa > 0.5 -> phase-interference type
        is_phase_type = res['kappa_median'] > 0.3 and res['kappa_n_above_0.5'] >= 20
        type_label = "PHASE-INTERF" if is_phase_type else "AMP-SUPPRESS"
        print(f"  {res['run_id']:>2d}  {res['kappa_mean']:>8.4f}  {res['kappa_median']:>9.4f}  "
              f"{res['kappa_n_above_0.5']:>6d}  {res['kappa_n_above_1.0']:>6d}  {type_label:>12s}")
    
    # --- Per-run classification ---
    phase_runs = [res['run_id'] for res in all_results
                  if res['kappa_median'] > 0.3 and res['kappa_n_above_0.5'] >= 20]
    amp_runs = [res['run_id'] for res in all_results
                if not (res['kappa_median'] > 0.3 and res['kappa_n_above_0.5'] >= 20)]
    
    print(f"\n--- Classification ---")
    print(f"  Phase-interference type ({len(phase_runs)} runs): {phase_runs}")
    print(f"  Amplitude-suppression type ({len(amp_runs)} runs): {amp_runs}")
    
    # --- Mean eta matrix across all runs ---
    all_eta = np.array([res['eta'] for res in all_results])
    mean_eta = np.mean(all_eta, axis=0)
    std_eta = np.std(all_eta, axis=0)
    
    print(f"\n--- Mean η matrix (mode × band) ---")
    print(f"{'':>8s}  {'θ-band':>8s}  {'α-band':>8s}  {'β-band':>8s}  {'γ-band':>8s}")
    mode_names = all_results[0]['labels']
    for i, m in enumerate(mode_names):
        print(f"  {m:>6s}  {mean_eta[i,0]:.4f}±{std_eta[i,0]:.3f}  "
              f"{mean_eta[i,1]:.4f}±{std_eta[i,1]:.3f}  "
              f"{mean_eta[i,2]:.4f}±{std_eta[i,2]:.3f}  "
              f"{mean_eta[i,3]:.4f}±{std_eta[i,3]:.3f}")
    
    # --- Save results to JSON ---
    save_path = "multiple_runs/summary"
    os.makedirs(save_path, exist_ok=True)
    
    save_data = {
        'n_runs': len(all_results),
        'run_ids': [res['run_id'] for res in all_results],
        'phase_interference_runs': phase_runs,
        'amplitude_suppression_runs': amp_runs,
        'mean_eta_diagonal': {
            'theta': f"{diag_eta[:,0].mean():.4f}±{diag_eta[:,0].std():.4f}",
            'alpha': f"{diag_eta[:,1].mean():.4f}±{diag_eta[:,1].std():.4f}",
            'beta': f"{diag_eta[:,2].mean():.4f}±{diag_eta[:,2].std():.4f}",
            'gamma': f"{diag_eta[:,3].mean():.4f}±{diag_eta[:,3].std():.4f}",
        },
        'mean_eta_matrix': mean_eta.tolist(),
        'std_eta_matrix': std_eta.tolist(),
        'per_run': [{
            'run_id': res['run_id'],
            'eta_diagonal': np.diag(res['eta']).tolist(),
            'eta_beta_alpha': float(res['eta'][res['labels'].index('beta'),
                                                list(band_names).index('alpha')]),
            'kappa_mean': res['kappa_mean'],
            'kappa_median': res['kappa_median'],
            'kappa_n_above_0.5': res['kappa_n_above_0.5'],
        } for res in all_results],
    }
    
    with open(f"{save_path}/phase_coherence_results.json", 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to {save_path}/phase_coherence_results.json")


if __name__ == '__main__':
    main()
