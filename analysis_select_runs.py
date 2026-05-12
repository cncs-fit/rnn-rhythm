"""
Phase 1: Select runs with high beta-gamma population overlap.
Reads precomputed Jaccard matrices from analysis_results.json of each run.
"""
import json
import os
import numpy as np

RUNS_DIR = "multiple_runs"
N_RUNS = 20
RHYTHM_NAMES = ['theta', 'alpha', 'beta', 'gamma']

def load_jaccard_matrices():
    """Load Jaccard matrices from all runs."""
    results = []
    for i in range(1, N_RUNS + 1):
        path = os.path.join(RUNS_DIR, str(i), "figures", "single", "analysis_results.json")
        with open(path) as f:
            data = json.load(f)
        jm = np.array(data['jaccard_matrix'])
        results.append({'run': i, 'jaccard_matrix': jm})
    return results

def main():
    results = load_jaccard_matrices()

    # Extract beta-gamma Jaccard coefficient (index [2,3])
    bg_jaccards = [(r['run'], r['jaccard_matrix'][2, 3]) for r in results]
    bg_jaccards.sort(key=lambda x: x[1], reverse=True)

    values = [v for _, v in bg_jaccards]
    print(f"Beta-Gamma Jaccard: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
    print(f"  min={np.min(values):.3f}, max={np.max(values):.3f}")
    print()

    print("All runs sorted by J(beta,gamma):")
    print(f"{'Run':>4s}  {'J(β,γ)':>8s}  {'J(θ,α)':>8s}  {'J(α,β)':>8s}  {'J(α,γ)':>8s}")
    print("-" * 50)
    for run_id, jbg in bg_jaccards:
        jm = next(r['jaccard_matrix'] for r in results if r['run'] == run_id)
        print(f"{run_id:4d}  {jbg:8.3f}  {jm[0,1]:8.3f}  {jm[1,2]:8.3f}  {jm[1,3]:8.3f}")

    # Suggest top runs
    print()
    print("Suggested runs (J(β,γ) > 0.5):")
    for run_id, jbg in bg_jaccards:
        if jbg > 0.5:
            print(f"  Run {run_id}: J(β,γ) = {jbg:.3f}")

if __name__ == "__main__":
    main()
