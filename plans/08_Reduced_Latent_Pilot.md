# Phase 8: scalar latent generation with the existing decoder

Authorization: approved October 10, 2026 by the user's "yep" response to this plan.

Status: complete. See [the results](08_Scalar_Pilot_Results.md); the scalar sampler
passed the validation target but showed no advantage over simple sampling.

Hypothesis: the source codes' effectively one-dimensional distribution is easier to
model with a scalar diffusion process, avoiding the original sampler's large-norm tail.
The current two-model design remains: a compact generator plus a frozen learned decoder.
This experiment tests a simpler interface, not a claim that more compression always helps.

Work:
1. Fit one-component PCA and coefficient mean/std using the 160 training codes only.
   Freeze these transforms and verify that projected/re-expanded validation codes,
   passed through the existing decoder, lose at most one percentage point of median
   accuracy relative to the existing autoencoder validation reconstructions. Stop if
   this gate fails; do not alter the decoder or add components without a revised plan.
2. Train one scalar epsilon-prediction DDPM: width 64, three residual blocks, timestep
   embedding width 32, same 200-step linear beta schedule 0.0005–0.1, seed 42, Adam
   0.001, batch size 32, at most 2,000 updates. No encoder or decoder updates.
3. Select the sampler checkpoint by median accuracy on five fixed validation seeds
   60001–60005 every 500 updates, earliest step on ties.
4. Compare 100 validation candidates per method using seeds 70001–70100: scalar diffusion,
   scalar Gaussian, empirical training-coefficient bootstrap, and the frozen original
   diagonal 128-coordinate Gaussian. Also retain the training-mean latent and the
   existing averaged-weight validation baseline. Equal stochastic candidate counts;
   select highest validation accuracy and lowest seed on ties for every method.
5. Report all accuracies, failure fractions, coefficient/expanded-latent ranges,
   nearest-training-latent distances, prediction disagreement, timing and memory.
   Do not clip, project sampled coefficients to their observed range, or reject samples.

Budget: one scalar training run, at most thirty minutes combined CPU training,
sampling, and validation evaluation. Preserve partial outputs if capped. No paid
compute, source retraining, additional seeds for training, official test evaluation,
held-out behavior, or stochastic decoder training.

Pilot quality target: median validation accuracy within two points of the current
94.67% training-source median and at least 80% of candidates inside that interval.
Also report whether scalar diffusion actually improves on scalar Gaussian/empirical
sampling; passing accuracy alone does not establish an advantage. No adaptive extension
of the budget or extra hyperparameter search after failures.

Deliverables: train-only transform tests, scalar generator/bundle and reproducible
commands, all-candidate CSVs and plots, and a results document in `plans/`. Any future
evaluation plan must state that MNIST official test outcomes have already been seen.
Preserve phase 6's frozen protocol/code and negative result; new work is exploratory.
