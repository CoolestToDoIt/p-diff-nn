# p-diff-nn

Research prototype for diffusion-generated MNIST classifier parameters. The fixed
classifier has 25,818 parameters. The classifier, parameter codec, data splits, and
baseline trainer, convolutional autoencoder, and latent export are implemented.
Diffusion work is planned.

The authorized source pilot is complete: baseline validation accuracy 94.30%, ten
fine-tuned checkpoints at 94.37–94.54%. See [results](plans/02_Source_Pilot_Results.md)
and [project plans and authorization status](plans/README.md).
Full source collection is also complete: 200 checkpoints in 2.45 minutes, with
94.67% training-branch and 94.65% validation-branch median validation accuracy.
See [collection results](plans/03_Source_Collection_Results.md). The 128-dimensional
autoencoder passed its reconstruction gate: median validation accuracy 94.14%, a
0.51 percentage-point loss. See [reconstruction results](plans/04_Autoencoder_Results.md).

## Setup and checks

Python 3.11 or later:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install -e '.[dev]'
python -m pytest
python -m p_diff.train_baseline --smoke --output artifacts/smoke
```

The smoke run uses synthetic random images to verify mechanics. Its accuracy is
not an MNIST result. Use a new output directory for each run.
The tested dependency snapshot comes from Python 3.12 on macOS ARM64; other
platforms may require different PyTorch wheels.

After authorizing the [source pilot](plans/02_Source_Checkpoint_Pilot.md), run the
baseline on MNIST training and validation images:

```sh
python -m p_diff.train_baseline --download --output artifacts/baseline
```

Collect the bounded two-branch pilot from its verified baseline:

```sh
python -m p_diff.train_sources --base artifacts/baseline --output artifacts/source-pilot
```

Pilot settings live in `configs/source_pilot.yaml`. The collector exports loadable
checkpoints, provenance, raw/normalized vectors, and validation diversity metrics.
Normalization fits only the training branch. Pilot validation is exploratory.

The authorized full collection uses the fixed eight training, one validation, and
one held-out branch split:

```sh
python -m p_diff.train_sources --config configs/source_full.yaml --output artifacts/source-full
python -m p_diff.audit_sources artifacts/source-full
```

Held-out behavioral metrics remain reserved. Each command requires a fresh collection
output directory; the audit checks saved checkpoint integrity and writes `audit.json`.

## Reconstruction and latent export

```sh
python -m p_diff.prepare_reconstruction
python -m p_diff.train_autoencoder
python -m p_diff.export_latents --checkpoint artifacts/autoencoder-pilot/128/best.pt
```

Preparation excludes the reserved held-out branch from the trainer's input artifact.
The bounded pilot selects the best checkpoint by validation reconstruction loss and
checks accuracy on all twenty validation-branch classifiers. It permits a single
256-dimensional retry only if the 128-dimensional gate fails. Export requires a
passing checkpoint and fits latent statistics on training rows only. Settings are
in `configs/autoencoder_pilot.yaml`; outputs default to `artifacts/autoencoder-pilot/`.

Commands refuse existing output paths; use CLI `--output` options and matching input
paths for repeated runs. Dataset and model binaries are ignored by git. The plans
folder includes results, CSV evidence, and a reconstruction plot. Reconstructed
variation is lower than source variation; passing accuracy is not evidence that
diffusion generation will outperform simple sampling.

Configuration lives in `configs/foundation.yaml`. Commands use CPU and never load
the official test set. Checkpoints contain weights and run settings; `manifest.json`
defines tensor ordering, and `metrics.json` records validation history and time.
