# Phase 4: bounded parameter autoencoder pilot

Authorization: user's October 6, 2026 instruction to keep going until a good git
commit milestone. Phase 3 audit passed. This phase ends at a tested reconstruction
result; diffusion training remains a separate substantial phase.

Status: complete. The 128-dimensional attempt passed; see
[results](04_Autoencoder_Results.md) and [next phase](05_Diffusion_Pilot.md).

Goal: compress the 25,818 classifier parameters into 128 latent coordinates while
preserving validation-branch classifier accuracy. This phase contains no diffusion.

Scope:
1. Inspect the authors' small-network and autoencoder implementation and licensing;
   record references and independently implement the architecture unless reuse is justified.
2. Build four strided 1D convolution blocks (channels 8, 16, 32, 32), latent projection,
   symmetric decoder, explicit padding/cropping, and serialized shape configuration.
3. Train only on the 160 training-branch vectors, using tensor-block-balanced
   reconstruction loss; mask padding. Start without noise augmentation.
4. Evaluate reconstruction weight error and classifier validation accuracy on all twenty
   validation-branch checkpoints. Never load held-out vectors into this training/evaluation
   workflow; use dataset row split filters and tests to enforce the restriction.
5. Select the best training checkpoint by validation reconstruction loss and verify the
   accuracy gate on it. If 128 dimensions fail, allow one 256-dimensional retry. Make
   any further changes a separately authorized plan.
6. Freeze a successful autoencoder, export deterministic training/validation latents,
   fit latent mean/std on training rows only, save configs/history and a reproducible
   reconstruction report, and propose a separate diffusion pilot plan.

Budget: CPU locally, at most 1,000 optimizer updates for each of the two allowed latent
sizes, batch size 16, Adam learning rate 0.001, seed 42. Check validation every 100
updates, stop early if nonfinite or clearly unstable, and cap combined training and
evaluation at 30 minutes. Run a short timing smoke first to assess this cap. No paid
compute, source retraining, official test evaluation, diffusion, or extra base seeds.
Persist partial histories/checkpoints if the time cap is reached.

Acceptance: output length is exactly 25,818; padding never contributes to the loss;
fresh decoded classifiers load and predict without updates; train-only normalization;
validation branch median original accuracy minus median reconstructed accuracy is
at most one percentage point. Also report the distribution of per-checkpoint accuracy
loss, block reconstruction error, elapsed time, and memory where available. Low parameter
error alone does not pass the gate. If neither allowed latent size passes, stop and
propose a revised reconstruction phase before diffusion.

Deliverables: autoencoder module/training/export commands and meaningful tests,
frozen checkpoint with architecture and source-dataset identity, training/validation
latent artifacts and statistics, CSV/plots of reconstruction metrics, a results document
in `plans/`, and a diffusion authorization plan only after the reconstruction gate passes.
