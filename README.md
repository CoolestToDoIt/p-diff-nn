# p-diff-nn

Research prototype for diffusion-generated MNIST classifier parameters. The fixed
classifier has 25,818 parameters. The classifier, parameter codec, data splits, and
baseline trainer, convolutional autoencoder, latent diffusion, standalone generation,
and locked official-test evaluation are implemented. The first single-seed prototype
is complete with a documented negative result for diffusion's reliability target.

The authorized source pilot is complete: baseline validation accuracy 94.30%, ten
fine-tuned checkpoints at 94.37–94.54%. See [results](plans/02_Source_Pilot_Results.md)
and [project plans and authorization status](plans/README.md).
Full source collection is also complete: 200 checkpoints in 2.45 minutes, with
94.67% training-branch and 94.65% validation-branch median validation accuracy.
See [collection results](plans/03_Source_Collection_Results.md). The 128-dimensional
autoencoder passed its reconstruction gate: median validation accuracy 94.14%, a
0.51 percentage-point loss. See [reconstruction results](plans/04_Autoencoder_Results.md).
The diffusion pilot completed: 93.55% median validation accuracy, versus 94.15% for
Gaussian sampling and 94.67% for averaged source weights. Diffusion's worst sample
was 55.75%; see [pilot results](plans/05_Diffusion_Results.md).

The locked 100-sample official-test evaluation is complete: diffusion median **94.08%**,
worst **31.24%**, and **77%** within two points of the 94.93% training-source median,
missing the required 80%. Gaussian median was **94.47%** with 100% in range; averaged
source weights scored **94.96%**. See [final results](plans/06_Locked_Evaluation_Results.md).
No training or selection used official test accuracy.

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

## Diffusion pilot and standalone generation

```sh
python -m p_diff.train_diffusion
python -m p_diff.generate_models --seed 30001 --output artifacts/generated-classifier.pt
python -m p_diff.evaluate --model artifacts/generated-classifier.pt
```

The pilot uses `configs/diffusion_pilot.yaml`, trains only on training-branch latents,
selects a checkpoint using five fixed validation generation seeds, and compares twenty
fresh diffusion candidates with twenty Gaussian candidates and two central baselines.
All candidate results and complete classifiers are retained. `generator.pt` under
`artifacts/diffusion-pilot/` contains a decoder-only standalone bundle; generation
loads no MNIST data, encoder, or source weights and applies no classifier updates.
`evaluate` loads only the saved classifier and fixed validation images. Pilot results
are exploratory; the current diffusion sampler has an unreliable tail.

Load a generated classifier:

```python
import torch
from p_diff.classifier import Classifier

checkpoint = torch.load("artifacts/generated-classifier.pt", weights_only=True)
model = Classifier().eval()
model.load_state_dict(checkpoint["state_dict"])
# images: float32 [batch, 1, 28, 28], scaled to [0, 1]
with torch.no_grad():
    labels = model(images).argmax(dim=1)
```

## Locked evaluation and example notebook

After preparing the existing source and generator artifacts, the evaluation workflow
freezes hashes and rules before any test access:

```sh
python -m p_diff.evaluation_protocol
python -m p_diff.locked_evaluation
```

The protocol and output paths must be new. Selections are saved before the official
test dataset can load; hashes and counts are checked throughout. Test accuracy never
chooses candidates. Model revisions after this run must disclose that test results
have already been seen. The report retains every candidate and source metric.

To reproduce the completed run without creating a different protocol:

```sh
python -m p_diff.locked_evaluation --protocol plans/results/06_Protocol.json --output artifacts/locked-evaluation-repeat
```

Open [the prediction notebook](notebooks/generated_classifier.ipynb) with the project
`.venv` kernel to load the validation-selected diffusion candidate and classify twelve
validation images without updates. A notebook frontend may need `ipykernel` installed
in that environment. Code cells were also verified by direct execution.

The saved standalone generator remains in `artifacts/diffusion-pilot/generator.pt`;
all 207 final classifiers are under `artifacts/locked-evaluation/models/`. Dataset and
model binaries stay local and ignored by git. Report evidence is committed under
`plans/results/` and `plans/figures/`. Optional follow-up work requires approval of
[the instability diagnosis plan](plans/07_Instability_Diagnosis.md).

Configuration lives in `configs/foundation.yaml`. Commands use CPU and never load
the official test set. Checkpoints contain weights and run settings; `manifest.json`
defines tensor ordering, and `metrics.json` records validation history and time.
