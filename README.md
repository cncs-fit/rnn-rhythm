# Multiple mechanisms of rhythm switching in RNNs with adaptive time constants

Code accompanying the paper

> Yamaguti Y, Nakamura S (2026) Multiple mechanisms of rhythm switching in
> recurrent neural networks with adaptive time constants. 
> (under review).

The arxiv link is https://arxiv.org/abs/XXXX.XXXXX

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
python analysis_phase_supfig.py      # Fig. A1 (kappa distribution)
```

`analysis_phase_coherence.py` writes
`multiple_runs/summary/phase_coherence_results.json`, which is consumed by
the other two scripts. The representative run for Fig. 6 is set inside
`analysis_phase_fig6.py` (`run_id = 2`); edit this line if you want to
inspect a different run. -->

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
└── make_fig4_intermode.py      # Fig. 4
```

## Citation

If you use this code, please cite:

```bibtex
@article{yamaguti2026rhythm,
  author  = {Yamaguti, Yutaka and Nakamura, Shota},
  title   = {Multiple mechanisms of rhythm switching in recurrent neural
             networks with adaptive time constants},
  journal={arXiv preprint arXiv:XXXXXXX},
  year    = {2026},
  note    = {under review}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Yutaka Yamaguti <y-yamaguchi@fit.ac.jp>
