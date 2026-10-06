# Phase 2: source checkpoint pilot

Authorization: approved by the user October 6, 2026. Applies only to this phase.

Status: complete. See [pilot results](02_Source_Pilot_Results.md) and the
[next authorization plan](03_Full_Source_Collection.md).

Goal: obtain a reproducible MNIST baseline and confirm that fine-tuning trajectories
have useful variation before collecting the full source dataset.

Work:
1. Download MNIST and run the five-epoch CPU baseline (seed 42); inspect validation accuracy.
2. Implement branch fine-tuning, checkpoint provenance, and parameter dataset export.
3. Pilot two branches, at most two epochs each, five spaced snapshots per branch.
4. Report validation accuracy, pairwise normalized weight distances, prediction disagreement,
   elapsed time, and suggested learning rate/checkpoint spacing.
5. Draft a separate authorization plan for ten branches and roughly 200 checkpoints,
   with a measured compute estimate and fixed 8/1/1 branch split.

Budget: one base seed, five baseline epochs plus four branch epochs, CPU by default;
stop training after 30 minutes total if unfinished and report the partial result.
No paid compute, autoencoder/diffusion training, or official test-set evaluation.
Persist seeds, optimizer settings, split indices, base-checkpoint identity, branch IDs,
steps, metrics, tensor manifest, and training-only normalization statistics.

Acceptance: exported checkpoints load into fresh classifiers without updates; provenance
is complete; train/validation image split is fixed; dataset normalization uses only
designated training branches. Pilot branches are exploratory and do not substitute
for the final held-out branches. If accuracy or diversity is inadequate, propose a
revised bounded pilot before expanding collection.

Future authorization phases: full source collection; autoencoder and reconstruction
gate (median validation loss of accuracy at most one percentage point); latent diffusion
pilot; locked evaluation and reporting. Create concrete plans as measurements become available.
