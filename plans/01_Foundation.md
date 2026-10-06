# Phase 1: repository and parameter foundations

Authorization: user's October 6, 2026 request to start implementation.

Scope: package setup, fixed 784–32–16–10 classifier, deterministic 50,000/10,000
training/validation split, ordered tensor manifest, strict parameter restoration,
training-only tensor-block normalization, bounded CPU baseline command, and tests.
Synthetic smoke runs check training and artifact mechanics without downloading MNIST.
No checkpoint collection, autoencoder, diffusion training, or official test evaluation.

Acceptance: 25,818 parameters; exact tensor and prediction round trip; numerically
stable normalization inverse; deterministic complete disjoint splits; smoke training
exports a checkpoint, manifest, metrics, and records synthetic data explicitly.

Reference inspected: the [authors' repository](https://github.com/NUS-HPC-AI-Lab/Neural-Network-Diffusion)
documents separate preparation, generation, and evaluation workflows. The code here
is independently implemented; no upstream code was copied. Inspect the upstream
small-network implementation and licensing before any later reuse.

Status: complete, October 6, 2026.

Verification: `python -m pytest -q` passed all three tests. The CPU synthetic smoke
run completed one optimizer step, exported `artifacts/smoke/baseline.pt`, manifest,
and metrics, and passed exact prediction and tensor round trips. Its random-image
accuracy is not research evidence. No MNIST data or official test images were loaded.
`git diff --check` passed. The tested Python 3.12 environment is pinned in
`requirements-tested.txt` (including torch 2.14.1 and torchvision 0.29.1).

Remaining limitations: only float32 trainable parameters are supported; this fixed
classifier has no buffers. CUDA/MPS execution, branch collection, checkpoint dataset
serialization, and real MNIST baseline quality have not been verified.
