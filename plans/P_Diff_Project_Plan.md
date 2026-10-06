# p-diff: Architecture and Project Plan

Status: Foundation, source collection, and parameter autoencoder pilot completed
October 6, 2026. The reconstruction gate passed. Diffusion and final evaluation remain
proposed. See [phase status and authorization plans](README.md).

## 1. Goal and limits

Build a small research prototype that generates the weights and biases of an MNIST classifier. A generated classifier should make predictions immediately, without gradient updates after generation.

The project tests this question: Can a diffusion model learn a useful distribution of trained weights and produce classifiers with accuracy close to the source models?

This does not remove training from the overall system. Source classifiers, the autoencoder, and the diffusion model still need training. Knowledge of MNIST reaches the generator through the source weights. We will not claim that this produces models for unseen tasks or that it is cheaper than ordinary training without measuring the full cost.

The paper's main setup generates selected normalization parameters and combines them with fixed trained parameters. It also includes full parameter generation for small MLPs and convolutional networks. Our main experiment follows the small-network idea, using our own architecture and evaluation protocol. This is a p-diff-inspired prototype, not an exact reproduction of the paper's reported numbers. [1]

## 2. Proposed scope

| Item | First version |
| --- | --- |
| Dataset | MNIST only |
| Generated model | One fixed MLP architecture |
| Parameters generated | All weights and biases |
| Compression | Small parameter autoencoder |
| Generator | Unconditional latent diffusion model |
| Interface | Command-line scripts and an example notebook |
| Output | Loadable PyTorch classifiers, metrics, plots, report |
| Main comparison | Source checkpoints, weight averaging, and simple sampling |

Later extensions: Fashion-MNIST, a small CNN, selected-layer generation, or faster sampling. Large models, arbitrary architectures, text-conditioned weight generation, and a web application are outside the first version.

Planning assumption: one developer, roughly six weeks, access to one GPU. All numerical settings below are starting points, not verified optimal settings.

## 3. System architecture

There are two separate workflows.

**Training:** MNIST → trained classifier checkpoints → flatten and normalize weights → autoencoder → latent dataset → diffusion training.

**Generation:** Gaussian noise → reverse diffusion → latent vector → decoder → undo normalization → restore tensors → new classifier.

Generation loads only the trained diffusion model, decoder, normalization statistics, and architecture manifest. It does not load MNIST, optimize the generated classifier, or copy source weights into it. Evaluation loads MNIST separately.

| Module | Responsibility | Saved output |
| --- | --- | --- |
| Dataset loader | Fix train/validation/test splits and preprocessing | Split indices and metadata |
| Source trainer | Train the classifier and collect compatible checkpoints | Checkpoints and trajectory IDs |
| Parameter codec | Flatten, normalize, and restore tensors in a fixed order | Vector dataset and tensor manifest |
| Autoencoder | Compress parameters and reconstruct them | Encoder, decoder, training history |
| Latent exporter | Encode source weights using the frozen encoder | Latent dataset and latent statistics |
| Diffusion trainer | Learn to predict noise added to latent vectors | Denoiser and diffusion configuration |
| Generator | Sample latents and export usable classifiers | Model files and generation seeds |
| Evaluator | Compare accuracy, reliability, novelty, and time | CSV results and plots |

## 4. Classifier and source checkpoint dataset

Use an MLP with layers `784 → 32 → 16 → 10`. Apply ReLU after the first two layers. The output contains ten class logits. This model has **25,818 trainable parameters**. Avoid batch normalization and dropout initially so there are no running statistics or stochastic inference behavior to manage.

Split MNIST's 60,000 training images into 50,000 training and 10,000 validation images using a fixed seed. Keep the official 10,000 test images untouched until the experiment settings and candidate selection rules are locked.

Train one base classifier using Adam and cross-entropy. Then make ten fine-tuning branches from that same checkpoint, using different minibatch orders and small learning rates. Collect about twenty spaced checkpoints per branch, giving about 200 parameter vectors.

**Why use a shared starting model?** Independently initialized networks can assign the same function to differently ordered hidden neurons. Mixing their flattened weights makes the distribution harder to learn. Nearby fine-tuning branches reduce this problem. This limits the initial claim to generation within a shared initialization family; it does not demonstrate generation across unrelated training runs.

Do not treat the 200 snapshots as 200 independent models. Record the branch, training step, seed, and metrics for every checkpoint. Pilot the checkpoint spacing and learning rate so snapshots have useful variation while retaining good accuracy.

Divide branches into eight training branches, one validation branch, and one held-out branch. Keep this split fixed for both the autoencoder and diffusion stages. No branch contributes checkpoints to multiple splits. This reduces adjacent-checkpoint leakage, although all branches still share the base model. Repeat the complete experiment with three base-model seeds if the compute budget allows.

## 5. Parameter codec

Store an explicit ordered manifest containing each parameter's name, shape, dtype, offset, and length. Flatten only according to this manifest. Reconstruct using the same manifest and reject architecture mismatches.

Normalize each tensor block using a mean and standard deviation estimated only from the training checkpoint vectors, with a minimum standard deviation to prevent division by zero. Retain these statistics for decoding. Compare against no normalization in a small pilot if reconstruction is unstable.

Before training the autoencoder, verify that flattening and restoring a checkpoint preserves every tensor and its predictions. Also verify that normalization and its inverse recover the original vector within floating-point tolerance.

## 6. Parameter autoencoder

Use a compact 1D convolutional encoder and decoder, reflecting the paper's use of 1D networks for parameter vectors. [1] Pad the 25,818-element vector to a compatible length and exclude padded values from the loss.

Starting design: four strided convolution blocks with channel widths 8, 16, 32, and 32, followed by a projection to a **128-dimensional latent vector**. The decoder reverses this structure and crops the output to the original length. Store the exact padding and output shapes in the configuration.

Train with parameter reconstruction error, giving each tensor block equal weight so the largest matrix does not dominate. Try small input and latent Gaussian noise as an augmentation; select noise levels on validation checkpoints. Noise is disabled during evaluation and ordinary latent export.

Evaluate both parameter error and classifier accuracy after reconstruction. Low weight error alone is insufficient. Freeze the autoencoder before training diffusion.

**Gate:** On the held-out validation branch, reconstructed classifiers should lose no more than one percentage point of median accuracy versus their original checkpoints. If this fails, increase the latent size to 256, reduce compression or augmentation, and fix reconstruction before proceeding.

## 7. Latent diffusion model

Encode the training checkpoint vectors with the frozen encoder. Standardize latent coordinates using training latents only; store the statistics. Reverse this transform before passing sampled latents to the decoder.

For this small prototype, use a residual MLP denoiser: 128 latent inputs, a sinusoidal timestep embedding, three hidden blocks of width 256, and 128 predicted noise outputs. This is a deliberate simplification of the paper's convolutional generator.

Start with a DDPM process using 200 noise steps and a fixed beta schedule. At each training step, choose a latent, sample a timestep and Gaussian noise, form the noisy latent, and minimize mean squared error between the sampled noise and predicted noise. [3]

At generation time, start from Gaussian noise and apply the reverse process. Decode the final latent, restore the complete parameter vector, and load it into a fresh classifier. Set the classifier to evaluation mode. Apply no classifier fine-tuning.

Use validation results to choose training duration. Start with a short pilot and a capped step budget rather than assuming diffusion loss alone predicts classifier quality. Log generated validation accuracy at fixed intervals.

## 8. Evaluation and success criteria

Generate 100 classifiers per final run using recorded random seeds. Report the full distribution; do not report only the best sample.

| Comparison | What it checks |
| --- | --- |
| Randomly initialized classifier | Chance-level reference |
| Source checkpoints | Performance of the training weight distribution |
| Autoencoder reconstructions | Accuracy lost through compression |
| Average source weights | Whether a simple shared-basin average suffices |
| Training latent mean decoded | Whether a single central latent already works |
| Diagonal Gaussian fitted to training latents | Whether simple latent sampling matches diffusion |
| Diffusion-generated classifiers | Main experimental result |

Use the same candidate count and validation-based selection rule for sampling methods. Select a deployment candidate using validation accuracy; test accuracy never determines selection or hyperparameters.

Report mean, median, standard deviation, worst accuracy, percentage within two percentage points of the source median, and the test accuracy of the validation-selected candidate. Report results per base-model seed and aggregate across seeds when available.

**Proposed success target:** at least 80% of generated classifiers achieve accuracy within two percentage points of the source-checkpoint median. Generated median accuracy should also fall within that range. These are project targets, not promised outcomes.

Measure nearest-source normalized weight distance and prediction disagreement with source models. Compare them with distances and disagreement among source checkpoints and with the held-out branch. These checks can expose copying or collapse, but they cannot prove a lack of memorization.

Measure source-training time, checkpoint collection, autoencoder training, diffusion training, sampling per model, peak device memory, and candidate evaluation. Compare generation with training this same classifier to similar accuracy. Separate one-time setup cost from per-model generation cost; include validation selection cost when reporting a selected model.

## 9. Implementation and deliverables

Use Python and PyTorch, with YAML configuration, NumPy, pandas, and Matplotlib. Pin the tested dependency versions and save seeds, dataset splits, architecture manifest, normalization statistics, and training settings.

Proposed repository modules: `classifier.py`, `parameters.py`, `autoencoder.py`, `diffusion.py`, `data.py`, and `metrics.py`.

Proposed scripts: `train_sources.py`, `build_parameter_dataset.py`, `train_autoencoder.py`, `export_latents.py`, `train_diffusion.py`, `generate_models.py`, and `evaluate.py`.

The final deliverable contains reproducible commands, a small saved generator bundle, sample generated classifiers, an evaluation CSV, accuracy and diversity plots, and a short report. An example notebook demonstrates loading a generated model and classifying images. A CPU smoke mode verifies the pipeline with a reduced budget.

Use the authors' implementation as a reference and inspect relevant small-network experiments before coding. Record any reused code and its license requirements. The official repository provides checkpoint preparation, generation, and evaluation workflows. [2]

## 10. Schedule and checkpoints

| Week | Work | Completion condition |
| --- | --- | --- |
| 1 | Dataset splits, classifier, parameter codec | Baseline trains; tensor round trip passes |
| 2 | Branch training and checkpoint collection | About 200 valid vectors with provenance |
| 3 | Autoencoder and reconstruction evaluation | Reconstruction accuracy gate passes |
| 4 | Latent diffusion pilot and generation | Fresh models load and predict without updates |
| 5 | Baselines, diversity checks, seed repeats | Locked evaluation produces complete metrics |
| 6 | Cost measurements, plots, documentation | Another person can reproduce the experiment |

This is a planning estimate. The checkpoint and reconstruction pilots determine the actual training budget. Budget for a single GPU with roughly 8–16 GB of memory as a starting assumption, then measure actual use. About 200 raw float32 parameter vectors occupy only about 21 MB; optimizer state and generator architecture will dominate memory use.

## 11. Risks and fallback scope

| Risk | Response |
| --- | --- |
| Checkpoints are nearly identical | Collect more spaced snapshots and modestly varied fine-tuning branches |
| Autoencoder changes predictions | Increase latent size or reduce compression; stop diffusion work until fixed |
| Generated weights perform poorly | Compare simple latent baselines, inspect normalization, and tune validation-selected settings |
| Small checkpoint dataset causes overfitting | Keep branch splits, restrict generator size, and increase source diversity carefully |
| Generated models repeat source behavior | Report similarity evidence and narrow the conclusions |
| Compute or time runs out | Generate only the final classifier layer while freezing the shared trained backbone |

The fallback changes the research claim: it produces a trained backbone with a generated head, not a fully generated classifier. Label it clearly and retain unsuccessful full-generation results. A valid negative result is still useful if the pipeline and evaluation are sound.

## 12. Review summary

The proposed first version generates **all 25,818 parameters of one MNIST MLP**, uses a **128-dimensional parameter latent**, and trains an **unconditional DDPM** over compatible checkpoint trajectories. It evaluates quality across many samples and compares diffusion against simpler methods. The scope is a reproducible research prototype with scripts and a notebook.

## References

1. Wang et al., Neural Network Diffusion: https://arxiv.org/html/2402.13144v1
2. Authors' implementation: https://github.com/NUS-HPC-AI-Lab/Neural-Network-Diffusion
3. Ho et al., Denoising Diffusion Probabilistic Models: https://arxiv.org/abs/2006.11239
