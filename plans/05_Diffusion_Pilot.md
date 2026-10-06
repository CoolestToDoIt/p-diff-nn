# Phase 5: bounded latent diffusion pilot

Authorization: approved October 6, 2026 by the request to do the next phase.

Status: complete. The narrow pilot target passed, but simple baselines performed
better. See [results](05_Diffusion_Results.md) and [locked evaluation plan](06_Locked_Evaluation.md).

Goal: test whether a learned unconditional latent sampler produces usable classifiers
and compare it with simpler latent baselines. Autoencoder reconstruction passed, but
validation weight variation was compressed substantially; sampling success is uncertain.

Scope: implement a residual MLP denoiser (128 inputs/outputs, sinusoidal timestep
embedding, three width-256 blocks), a 200-step DDPM with an explicit fixed beta schedule,
noise-prediction training, and seeded sampling. Freeze the autoencoder and latent
normalization. Train only on the 160 training latents, never the validation or held-out
branch. Generated models must load into fresh classifiers without optimization.

Pilot budget: seed 42, Adam 0.001, batch size 32, at most 2,000 optimizer updates and
30 minutes combined CPU training/sampling/evaluation. Check five fixed generation seeds
on validation images every 500 updates, choose the training checkpoint using their
median validation accuracy, and label this as exploratory tuning. Record full seed
lists, schedule, configuration, costs, artifact identities, and partial outputs if capped.
No paid compute, source retraining, official test evaluation, or held-out evaluation.

At the selected checkpoint generate twenty candidates with fresh, predetermined seeds
for each sampling method: diffusion and diagonal Gaussian fitted on training latents.
Also evaluate decoded training latent mean and average training source weights. Keep
candidate counts and validation selection rules the same for stochastic methods.
Report every sample, median/mean/worst accuracy, fraction within two points of the
training-source median, distance to source weights, validation prediction disagreement,
and per-model time. Retain all failures. These are pilot results, not locked final results.

Acceptance: math/unit checks for forward noise and reverse sampling, reproducible seeded
generation, usable complete classifier files, explicit train-only statistics, and a
complete baseline comparison. The proposed quality target is generated median accuracy
within two points of the 94.67% training-source median, with at least 80% of candidates
within that range. Failure does not justify silently extending the step budget or
changing reconstruction; propose a separate revision if needed.

Deliverables: diffusion/generation/evaluation modules and commands, tests, frozen
generator bundle, sample classifier files, CSV/plots and results in `plans/`, and a
locked-evaluation authorization plan. Final 100-sample experiments, official test
metrics, multiple base seeds, and publication claims remain future phases.
