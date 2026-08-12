# Multiple mechanisms of rhythm switching in RNNs with adaptive time constants

Code accompanying the paper

> Yamaguti Y, Nakamura S (2026) Multiple mechanisms of rhythm switching in
> recurrent neural networks with adaptive time constants. 
> (under review).

The arxiv link is https://arxiv.org/abs/2605.14388

This repository provides the training and analysis pipeline used to produce
all figures in the paper. A leaky integrator RNN with neuron-specific
learnable time constants is trained on a four-band rhythm-switching task
(theta, alpha, beta, gamma) and analyzed across 20 independently trained
networks.

## Requirements

- Python 3.10
- TensorFlow 2.10
- GPU recommended (the paper used an NVIDIA RTX 5090; training also runs on
  CPU but is substantially slower)

Other dependencies are listed in `requirements.txt`.

## Setup

In case of conda environment:
```bash
conda create -n rhythm python=3.10
conda activate rhythm
pip install -r requirements.txt
```

## Reproducing the paper results

### 1. Train 20 networks

```bash
bash run_multiple.sh 20 42 1
```

This runs `train_once.py` 20 times with seeds 43, 44, ..., 62 and collects
the trained models under `multiple_runs/<i>/results/`.

To run a single training in isolation:

```bash
python train_once.py --seed 42
```

The trained model is written to `results/` and should be moved to
`multiple_runs/<i>/results/` if you want it picked up by the analysis
scripts below.

### 2. Aggregate per-run metrics across all 20 runs

```bash
python analysis_multiple.py
```

Computes amplitude / synchronization statistics, time-constant correlations,
and inter-mode similarity matrices for each run, and writes summary CSV /
JSON files to `multiple_runs/summary/`. Required for Fig. 4 and Table 1.
<!-- 
### 3. Generate per-run figures (Figs. 1--3, representative run)

```bash
python make_fig1_task_overview.py --run_id 1
python make_fig2_amp_tau.py        --run_id 1
python make_fig3_sync.py           --run_id 1
```

Each script reads the trained model under `multiple_runs/<run_id>/results/`
and writes a PDF figure to `multiple_runs/<run_id>/figures/`.

### 4. Generate Fig. 4 (inter-mode population similarity)

```bash
python make_fig4_intermode.py
```

Reads the multi-run summary produced in step 2.

### 5. Fig. 5 (baseline shift and fixed-point analysis)

The fixed-point analysis is performed per run. To select a representative
run with high beta--gamma Jaccard overlap:

```bash
python analysis_select_runs.py
```

Then run the fixed-point and Jacobian analysis for the selected run:

```bash
python analysis_baseline_fp.py --runs <run_id>
```

`<run_id>` is the run id reported by `analysis_select_runs.py`
(e.g. `--runs 19`).

### 6. Fig. 6 and Appendix A (phase coherence analysis)

```bash
python analysis_phase_coherence.py   # 20-run screening
python analysis_phase_fig6.py        # Fig. 6 (representative run)
python analysis_phase_supfig.py      # Fig. 9 (kappa distribution, Appendix A)
```

`analysis_phase_coherence.py` writes
`multiple_runs/summary/phase_coherence_results.json`, which is consumed by
the other two scripts. The representative run for Fig. 6 is set inside
`analysis_phase_fig6.py` (`run_id = 2`); edit this line if you want to
inspect a different run. -->

### 7. Additional experiments and analyses (revised version)

The following scripts implement the analyses added during revision. Except for
the extra training (`train_reinit.py` / `run_reinit.sh`), they are
self-contained: they read the trained model weights directly from the `.h5`
checkpoints (via a numpy re-implementation of the forward model) and do not
require TensorFlow.

```bash
# Causal silencing of time-constant-ranked subpopulations (Sect. 3.6; Figs. 10, 11)
python analysis_ablation.py --num_runs 20

# Recurrent connectivity, effective time constants, gain modulation (Sect. 3.8; Fig. 7)
python analysis_connectivity.py

# Inter-mode population sharing vs. learned median time constant (Sect. 3.4; Fig. 12)
python analysis_outlier_tau.py

# Time-constant-initialization control (Sect. 3.9; Figs. 13, 14):
#   train 10 log-uniform (tau in 5-300 ms) + 10 uniform-200 ms networks, then compare
bash run_reinit.sh 4 1e-5
python analysis_reinit_compare.py

# Stability of the tau-amplitude correlations between the 1e-5 and 1e-6 checkpoints
python analysis_stability_check.py
```

`run_reinit.sh` writes the additional networks under
`multiple_runs_reinit/{loguniform,const200}/<i>/`; `analysis_reinit_compare.py`
compares them against the baseline (50 ms) networks in `multiple_runs/`.
Outputs are written to `ablation_results/`, `connectivity_results/`,
`outlier_analysis/`, and `reinit_compare/`.

## Repository layout

```
.
├── train_once.py               # one training run (loss = band power ratio)
├── run_multiple.sh             # wrapper that runs train_once.py N times
├── function.py                 # shared signal-processing utilities
├── dataset_generator.py        # rhythm-switching task generator
├── mycallbacks.py              # Keras training callbacks
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   └── leaky_rnn_model.py      # ATCLRNN cell with learnable time constants
├── analysis_multiple.py        # cross-run aggregation
├── analysis_select_runs.py     # helper for picking representative runs
├── analysis_baseline_fp.py     # Fig. 5: fixed-point and Jacobian analysis
├── analysis_phase_coherence.py # 20-run phase classification
├── analysis_phase_fig6.py      # Fig. 6
├── analysis_phase_supfig.py    # Fig. A1
├── make_fig1_task_overview.py  # Fig. 1
├── make_fig2_amp_tau.py        # Fig. 2
├── make_fig3_sync.py           # Fig. 3
├── make_fig4_intermode.py      # Fig. 4 (inter-mode Spearman correlation + Jaccard)
│
│   # --- added during revision (self-contained; numpy, no TensorFlow) ---
├── train_reinit.py             # training with alternative time-constant initializations
├── run_reinit.sh               # wrapper for the initialization-control runs
├── analysis_ablation.py        # Sect. 3.6 / Figs. 9, 10: causal silencing (zero / freeze)
├── analysis_connectivity.py    # Sect. 3.8 / Fig. 12: connectivity, effective tau, gain modulation
├── analysis_outlier_tau.py     # Sect. 3.4 / Fig. 11: inter-mode sharing vs median tau
├── analysis_reinit_compare.py  # Sect. 3.9 / Figs. 13, 14: initialization robustness
└── analysis_stability_check.py # 1e-5 vs 1e-6 checkpoint stability check
```

## Citation

If you use this code, please cite:

```bibtex
@article{yamaguti2026rhythm,
  author  = {Yamaguti, Yutaka and Nakamura, Shota},
  title   = {Multiple mechanisms of rhythm switching in recurrent neural
             networks with adaptive time constants},
  journal={arXiv preprint arXiv:2605.14388},
  doi = {10.48550/arXiv.2605.14388},
  year    = {2026},
  note    = {under review}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Yutaka Yamaguti <y-yamaguchi@fit.ac.jp>
