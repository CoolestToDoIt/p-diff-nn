# Phase 6: locked single-seed evaluation

Authorization: approved October 10, 2026 by the request to keep going after review of
this next-phase plan. Covers official MNIST test images and held-out branch evaluation;
no training or hyperparameter changes are authorized.

Status: complete October 10, 2026. Diffusion missed the locked 80% reliability target;
see [results](06_Locked_Evaluation_Results.md). All planned comparisons and deliverables
are complete. An [optional diagnosis](07_Instability_Diagnosis.md) remains unapproved.

Goal: characterize the frozen pilot generator with a larger sample count and final
test evidence, including negative results. The pilot met its narrow target but lost
to Gaussian sampling and averaged weights; do not present diffusion as an improvement.

Before accessing test images, save an immutable evaluation protocol containing hashes
of the selected step-2,000 generator, source dataset, image splits, and latent/parameter
statistics, plus seed lists, candidate-selection rules, metric definitions, and sample
counts. Validate these hashes on every run. Make no model or hyperparameter changes
after test access within this phase.

Frozen settings: current 128-dimensional decoder and 200-step DDPM with linear betas
0.0005–0.1; no training or fine-tuning. Generate 100 diffusion and 100 diagonal Gaussian
classifiers using seeds 40001–40100 for each method. Also evaluate the training latent
mean, average training source weights, and five random classifiers (seeds 50001–50005).
The stochastic methods have equal counts and identical selection rules. Select their
deployment candidates using highest validation accuracy, lowest seed on ties; write
those selections before official test evaluation. Do not select from test accuracy.

Then evaluate all candidates on the official 10,000 test images. Report per-method
mean, median, population standard deviation, worst accuracy, fraction within two points
of the training-source median test accuracy, and test accuracy of the validation-selected
candidate. Apply the existing 80%/two-point target to test results without retuning.
Evaluate all 200 source checkpoints, reporting training/validation/held-out branches
separately, and reconstruct the twenty held-out source checkpoints with the frozen
autoencoder to check generalization. All source reference thresholds for the main
quality target use the 160 training-source checkpoints; report the others as separate
comparisons. Preserve raw per-checkpoint results and avoid treating trajectory snapshots
as independent experimental repeats.

Novelty checks: nearest-training-source normalized weight RMS and prediction disagreement
for generated candidates, compared with source-to-other-source and held-out-to-training
references. Exclude self-comparisons. Report candidate-to-candidate disagreement and
collapse evidence; do not equate bad classifiers with useful diversity or claim that
these checks disprove memorization. No official test evidence informs model selection.

Cost reporting: distinguish source setup, autoencoder/diffusion setup, bundle load,
sampling, candidate validation/selection, final evaluation, file export, and memory.
Use existing measured source-training history as context, explicitly noting that no
controlled matching-accuracy cost comparison has been performed.

Budget: one base seed, frozen artifacts, at most 30 minutes CPU evaluation, no paid
compute, new training, or seed repeats. Save completed candidate outputs and partial
metrics if capped; never silently shrink the prespecified sample count.

Deliverables: locked protocol artifact, reusable evaluation commands and meaningful
tests, full CSVs and plots, model selections, a brief report, an example notebook for
loading a generated classifier, and reproducibility instructions. Commit report evidence;
keep dataset/model binaries ignored. A valid negative outcome completes this phase.
Future training revisions or three-base-seed repeats require a separate authorization plan.
