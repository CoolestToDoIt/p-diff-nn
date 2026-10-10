# Phase 9 proposal: diagnose representation and decoder sensitivity

Authorization: awaiting user approval. No work under this phase has started.

Phase 8 stabilized scalar generation, but mean-latent, Gaussian and bootstrap
classifiers all remain near 94.15% validation accuracy, versus 94.67% for weight
averaging. Before training an additional expansion model, determine what information
the current encoder/decoder preserves and how much classifier behavior changes
within the observed latent range.

This is exploratory follow-up: MNIST official test outcomes were already seen in
phase 6. Do not access the official test dataset or held-out branch behavior.
Preserve the original frozen evaluation and phase 8's fixed results.

Scope:
1. Use the existing 160 training and 20 validation reconstruction inputs and frozen
   encoder/decoder. No full 200-row source artifact or new source training.
2. Fit PCA separately on training normalized weights and training standardized codes.
   Compare explained variance and rank without fitting on validation checkpoints.
3. Compare per-tensor reconstruction errors, reconstructed variation and pairwise
   prediction diversity with original source classifiers and weight averaging.
   Report train and validation groups separately; include reconstruction accuracy
   and changes in predictions, without selecting new generated candidates.
4. Probe decoder sensitivity at a predeclared grid of 21 equally spaced scalar
   coefficients from the observed training minimum to maximum, plus the scalar
   mean. Measure output-weight variation and validation predictions. No range
   extension, sampler tuning, checkpoint selection, or learned expansion changes.
5. Produce scripts, tests where needed, all-probe evidence, plots and a result report
   in `plans/`. Recommend a concrete bounded reconstruction-training pilot only
   if the diagnosis supports it; that pilot needs a separate authorization plan.

Budget: at most 30 minutes CPU for diagnosis, no paid compute or dependency additions.
No training updates, official test evaluations, stochastic decoder, new generation
seeds, or expansion-model implementation. Stop and preserve partial evidence at cap.

Decision: identify whether the low-dimensional code reflects low-dimensional source
variation, loss of useful information in the encoder, weak decoder sensitivity, or
an unresolved combination. Correlation and PCA rank alone do not establish a cause.
