# Phase 10 proposal: reconstruct residual weights around a fixed mean

Authorization: awaiting user approval. No implementation or training has started.

Phase 9 found that 97.3% of balanced validation reconstruction error comes from a
wrong shared parameter template; a constant training-weight mean has 33.6 times
lower MSE. Test a fixed mean anchor plus learned parameter differences before
training another generator or stochastic expansion model.

This is exploratory work after phase 6's official test outcomes were seen. Do not
access the official test dataset or held-out branch behavior. Preserve all existing
frozen source, autoencoder, generator and evaluation artifacts and results.

Authorized scope upon approval:
1. Use only the existing 160 training / 20 validation reconstruction inputs. Fit a
   per-coordinate mean parameter vector on training rows. Subtract that anchor and
   fit one residual RMS scale per tensor using training rows, with a 1e-6 floor.
   Decode as `training_mean + tensor_scale * decoded_normalized_residual`. Store
   the anchor, scales, provenance and classifier manifest in a standalone decoder.
   Validation rows must not affect the anchor, scales, PCA, or training updates.
2. Fit fixed PCA reconstruction baselines with 1, 8 and 64 components on training
   residuals only. Include the constant mean and original autoencoder reconstructions.
   No sampling from these representations and no adaptive component choices.
3. Train exactly one residual autoencoder with the existing convolutional architecture,
   latent size 128, seed 42, Adam 0.001, batch size 16, and at most 1,000 updates.
   Initialize the decoder's final output layer to zero so update zero exactly
   reconstructs the training-weight mean. Do not retrain source classifiers.
4. Record equal-tensor-weighted normalized residual MSE on validation checkpoints
   at update zero and every 100 updates. Select minimum validation MSE, earliest
   update on ties. No classifier-accuracy checkpoint selection, retries, changed
   architectures, additional seeds, or hyperparameter search.
5. Evaluate original, constant-mean, PCA and selected neural reconstructions on the
   fixed validation image split. Report each training/validation checkpoint, per-tensor
   errors, spectra, source-versus-reconstructed weight variation and prediction
   disagreement. Preserve all results if the neural model loses to a simple baseline.

Gates for exporting a new frozen learned representation:
- Median validation-branch reconstruction accuracy loss versus original checkpoints
  is at most 0.25 percentage points (original median 94.65%).
- Validation centered reconstruction error is at most 50% of original source variance,
  measured in the original normalized weight coordinates used in phase 9. Output
  variance alone is insufficient evidence of preserved information.
- Transform provenance, dimensions, roundtrips and train-only fitting checks pass.

If both quality gates pass, export deterministic train/validation codes and the
frozen standalone decoder with training-only code statistics. If either fails,
stop with the report and saved partial/selected checkpoint. Do not train a diffusion
sampler, add a stochastic decoder, extend the budget or relax a gate.

Budget: one training run, at most 30 minutes combined CPU preprocessing, training,
baseline reconstruction and validation evaluation; no paid compute or new dependencies.
Preserve partial outputs at the cap. Implement tests for train-only residual fitting,
anchored decoding and artifact compatibility, and write evidence/plots/report in `plans/`.

Decision: determine whether mean anchoring improves reconstruction and preserves useful
variation, and whether a learned residual decoder offers value beyond linear PCA.
Any subsequent parameter-generation pilot requires a separate authorization plan.
