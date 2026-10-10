# Phase 7: validation-only diffusion instability diagnosis

Authorization: proposed, awaiting user approval. The first prototype is complete;
this is an optional investigation, not required to finish the locked evaluation.

Question: which latent characteristics explain the diffusion sampler's weak tail?
The locked run's 77% reliability missed the 80% target, while Gaussian sampling
and averaged source weights were more reliable. Do not change this result.

Scope: load the frozen generator and the training/validation latent artifact only.
Recreate standardized diffusion and Gaussian latents for the existing locked seeds
40001–40100. Use only their already-saved validation accuracies for behavioral
comparisons. Inspect latent norm, coordinate extremes, distance to training latents,
and residual from a PCA subspace fitted on training rows only. Compare training,
validation, Gaussian, and diffusion latents. Inspect intermediate reverse-process
trajectories for five predetermined seeds 40001–40005 without changing sampling.

Budget: at most fifteen minutes CPU, no optimization, new candidate selection,
official test evaluation, held-out branch access, source retraining, or paid compute.
Do not clip latents, alter schedules, or add rejection sampling within this diagnosis.
Report correlations as exploratory observations, not causal explanations.

Deliverables: reproducible diagnostic command, train-only PCA tests, latent/trajectory
plots and CSV evidence, a brief diagnosis in `plans/`, and a separately authorized
training/sampling revision plan only if there is a concrete hypothesis to test.

Any subsequent revision should select settings using validation only, retain the
current baselines, and clearly disclose that MNIST test results have already been
seen. It cannot be presented as a fresh untouched-test replication.
