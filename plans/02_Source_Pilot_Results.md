# Source checkpoint pilot results

Completed October 6, 2026 under phase 2 authorization. CPU, one thread, Python 3.12,
seed 42, fixed 50,000/10,000 MNIST image split. Official test archives were downloaded
because torchvision expects the complete archive set, but test images were never loaded
or evaluated. All archive MD5 checks passed against torchvision's published resources.

Baseline: Adam, learning rate 0.001, batch size 128, five epochs. Validation accuracy
by epoch: 89.42%, 92.09%, 93.01%, 93.92%, 94.30%. Baseline time: 8.55 seconds.

Branches: both reset to the same baseline weights, fresh Adam optimizers, learning rate
0.0001, batch size 128, seeds 142 and 242, two epochs each. Five checkpoints per branch
at steps 157, 313, 470, 626, 782 (about 156 updates apart). Branch times: 4.92 and
5.01 seconds, including checkpoint validation. Baseline plus collection: 18.51 seconds,
well within the 1,800-second cap; downloads and implementation time excluded.

| Metric | Result |
| --- | --- |
| Snapshot count | 10 |
| Validation accuracy range | 94.37–94.54% |
| Mean / median validation accuracy | 94.469% / 94.460% |
| Within-branch normalized weight RMS distance, mean (range) | 0.01411 (0.00858–0.02440) |
| Across-branch normalized weight RMS distance, mean (range) | 0.01443 (0.00753–0.02484) |
| Within-branch prediction disagreement, mean (range) | 0.5815% (0.35–0.82%) |
| Across-branch prediction disagreement, mean (range) | 0.5600% (0.29–0.96%) |

Distances use tensor-block normalization fitted on branch 0 alone. Branch 1 is the
exploratory validation branch. Prediction disagreement uses all 10,000 validation
images. The 45 pairs are correlated comparisons, not independent trials.

Interpretation: accuracy stays stable and checkpoints differ in weights and predictions,
but variation is modest. Across-branch disagreement is similar to within-branch
disagreement. This supports a cautious source collection, not a claim that the eventual
diffusion generator will work. Keep learning rate 0.0001 and approximately 156-update
spacing initially; extend trajectories to eight epochs for twenty snapshots. Recheck
diversity and accuracy before any autoencoder work; a central-weight baseline may suffice.

Artifacts (local, ignored by git):
- `artifacts/pilot-baseline/`: baseline weights, image split, manifest, validation history.
- `artifacts/source-pilot/`: ten classifiers, provenance, split, manifest,
  `parameters.pt` with raw/normalized vectors and training-only statistics, pair metrics.

Verification: each collected checkpoint was restored into a fresh classifier and checked
for exact predictions before export; four tests pass, including validation leakage
prevention. The codec normalization inverse passes floating-point tolerance. No source
classifier was optimized after restoration for evaluation.

Reproduce with fresh output directories:

```sh
PYTHONPATH=src .venv/bin/python -m p_diff.train_baseline --download --output artifacts/pilot-baseline
PYTHONPATH=src .venv/bin/python -m p_diff.train_sources
.venv/bin/python -m pytest -q
```

If Python HTTPS certificate validation fails, repair its CA configuration or use the
system HTTPS client to download and extract the torchvision MNIST archives; do not
disable certificate verification.
