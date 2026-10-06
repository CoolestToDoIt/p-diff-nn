# Full source collection results

Completed October 6, 2026 under phase 3 authorization. CPU, one thread, deterministic
operations, seed-42 baseline shared by ten independently restarted fine-tuning branches.
Configuration: `configs/source_full.yaml`; Adam 0.0001, batch size 128, eight epochs,
twenty snapshots per branch, roughly 156 updates apart. Optimizer defaults, seeds,
base checkpoint SHA256, image split SHA256, and steps are saved in every checkpoint.

| Result | Value |
| --- | --- |
| Checkpoints | 200: 160 training, 20 validation, 20 held-out |
| Collection and pair analysis time | 146.92 seconds (2.45 minutes) |
| Artifact directory size before audit | 66.92 MB |
| Training-branch validation accuracy, mean / median | 94.6516% / 94.67% |
| Training-branch validation accuracy range | 94.37–94.89% |
| Validation-branch validation accuracy, mean / median | 94.6545% / 94.65% |
| Validation-branch accuracy range | 94.39–94.89% |
| Within-branch normalized weight RMS, mean (range) | 0.03802 (0.00795–0.09985) |
| Across-branch normalized weight RMS, mean (range) | 0.03766 (0.00720–0.10008) |
| Within-branch prediction disagreement, mean (range) | 0.6641% (0.21–1.40%) |
| Across-branch prediction disagreement, mean (range) | 0.6521% (0.12–1.46%) |

Pair statistics cover the nine training/validation branches only: 1,710 within-branch
pairs and 14,400 across-branch pairs. These correlated comparisons do not establish
independent samples or broad model diversity. Tensor-block normalization is fitted
only on the 160 training vectors. Official test images and held-out behavioral metrics
were not evaluated; held-out integrity was checked using tensor restoration only.

All collection gates passed: every saved checkpoint loads into the fixed classifier;
its vector exactly matches the dataset; all vectors are finite and distinct; image
splits are disjoint and complete; branch assignments match the frozen configuration;
provenance and split hashes agree; normalization can be exactly refitted from training
rows and inverted within tolerance. Training and validation medians exceed the 93.30%
accuracy floor. Five tests pass, including validation/held-out normalization exclusion;
`git diff --check` passes.

Interpretation: longer trajectories increased mean normalized weight separation from
the two-epoch pilot, while prediction differences remain small. Source quality is stable.
There is enough measured variation to try bounded reconstruction, but no evidence yet
that diffusion will outperform weight averaging or a central latent. Preserve these
simple baselines in later evaluation. Do not claim 200 independent models.

Local artifacts (ignored by git): `artifacts/source-full/` contains 200 checkpoint files,
`inventory.csv`, `provenance.json`, frozen `run.json`, `splits.pt`, `manifest.json`,
`parameters.pt`, `metrics.json`, and `audit.json`. Raw and normalized vectors both have
shape `[200, 25818]`; normalization fit rows are exactly 0–159. Pilot artifacts remain
separate. Artifact identities are recorded in `run.json`; held-out records have null
validation accuracy and do not appear in pair metrics.

Reproduction after setup (use a fresh collection directory):

```sh
PYTHONPATH=src .venv/bin/python -m p_diff.train_sources --config configs/source_full.yaml --output artifacts/source-full
PYTHONPATH=src .venv/bin/python -m p_diff.audit_sources artifacts/source-full
.venv/bin/python -m pytest -q
```

Next: review [the autoencoder pilot plan](04_Autoencoder_Pilot.md). No autoencoder
implementation or training is authorized by phase 3.
