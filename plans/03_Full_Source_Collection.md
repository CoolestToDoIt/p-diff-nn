# Phase 3: full source checkpoint collection

Authorization: approved October 6, 2026 ("go for it"). Covers collection and analysis only.

Status: complete; all gates passed. See [results](03_Source_Collection_Results.md)
and [next authorization plan](04_Autoencoder_Pilot.md).

Goal: build roughly 200 compatible parameter vectors with a fixed branch-level split,
then decide whether the source distribution is suitable for autoencoder experiments.

Use the verified seed-42 baseline from phase 2. Independently restart ten branches at
its exact saved weights, fresh Adam optimizer, learning rate 0.0001, batch size 128,
eight epochs, twenty snapshots per branch at approximately 156-update intervals.
Pilot snapshots remain exploratory artifacts and are not merged into the final dataset.

Freeze branch assignments before training:

| Branch IDs | Seeds | Split | Expected vectors |
| --- | --- | --- | --- |
| 0–7 | 142, 242, 342, 442, 542, 642, 742, 842 | train | 160 |
| 8 | 942 | validation | 20 |
| 9 | 1042 | held_out | 20 |

These IDs denote checkpoint trajectories, not image splits. All source training uses
only the 50,000 training images. Fit tensor-block normalization on the 160 training
vectors only. Do not use held-out branch accuracy or weights to choose hyperparameters;
reserve held-out analysis for locked evaluation. Validate held-out checkpoint integrity
without publishing validation-based tuning conclusions from that branch.

Deliverables: source checkpoints, full provenance, split indices, base SHA256 identity,
raw and normalized vector dataset, manifest/statistics, CSV inventory, training/validation
branch accuracy and diversity report, measured time/storage, and a separate autoencoder
authorization plan if the gates pass. Adjust the collector to keep held-out behavioral
metrics reserved and record optimizer defaults and split identity explicitly.

Estimated CPU collection time: roughly 2–4 minutes, based on 1.65 seconds per baseline
epoch, two pilot branches taking about five seconds each with five validation passes,
and scaling to eighty training epochs plus 180 training/validation checkpoint evaluations.
This is an extrapolation; machine load and storage can change it. Raw vectors occupy
about 21 MB; normalized vectors and checkpoints add similar amounts.

Hard budget: 30 minutes CPU collection, one base seed, no paid compute or extra dataset.
Stop at the cap and retain completed snapshots. No autoencoder/diffusion training,
official test evaluation, or three-seed repeat is authorized by this phase.

Gates: all checkpoints load and round-trip; complete provenance; no cross-split branches;
statistics fitted only on training vectors; finite, distinct weight vectors; training
and validation branch median accuracy at least 93.30% (within one point of the baseline).
Report pairwise distances and disagreement instead of assuming 200 snapshots are
independent. If accuracy deteriorates or near-duplicate behavior dominates, stop before
autoencoder training and propose a bounded revision. Do not widen learning rates or
add branches automatically.
