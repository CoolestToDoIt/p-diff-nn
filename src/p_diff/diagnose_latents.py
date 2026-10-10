"""Train-only PCA and validation-only diagnostics of frozen latent sampling."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

import torch

from .evaluation_protocol import file_hash, write_exclusive
from .generate_models import Generator
from .train_diffusion import training_latents


def fit_pca(training, retained_variance=.99):
    if training.ndim != 2 or len(training) < 2 or not torch.isfinite(training).all():
        raise ValueError("Need finite training rows for PCA")
    if not 0 < retained_variance <= 1:
        raise ValueError("Retained variance must be in (0,1]")
    training = training.double()
    mean = training.mean(0)
    _, singular, basis = torch.linalg.svd(training-mean, full_matrices=False)
    energy = singular.square()
    if energy.sum() <= 0:
        raise ValueError("PCA requires nonzero training variance")
    cumulative = (energy/energy.sum()).cumsum(0)
    rank = min(int(torch.searchsorted(cumulative, torch.tensor(retained_variance, dtype=torch.float64)))+1, len(basis))
    return {"mean": mean, "basis": basis[:rank], "rank": rank, "cumulative_variance": cumulative,
            "singular_values": singular, "retained_variance": retained_variance}


def describe_latent(vector, training, pca, exclude_row=None):
    centered = vector.double()-pca["mean"]
    reconstructed = (centered @ pca["basis"].T) @ pca["basis"]
    distances = (training-vector[None]).square().mean(1).sqrt()
    if exclude_row is not None:
        distances[exclude_row] = float("inf")
    return {"norm": vector.norm().item(), "max_abs_coordinate": vector.abs().max().item(),
            "nearest_training_latent_rms": distances.min().item(),
            "pca_residual_rms": (centered-reconstructed).square().mean().sqrt().item(),
            "pca_score_norm": (centered @ pca["basis"].T).norm().item()}


def trace_sampling(generator, seed):
    """Observe the unchanged sampler; never project, clip, or alter reverse steps."""
    diffusion = generator.diffusion
    rng = torch.Generator().manual_seed(seed)
    vector = torch.randn(1, generator.denoiser.config.latent_size, generator=rng)
    snapshots = [(0, vector[0].clone())]
    with torch.no_grad():
        for index in reversed(range(diffusion.config.steps)):
            times = torch.full((1,), index, dtype=torch.long)
            noise = torch.randn(vector.shape, generator=rng) if index else torch.zeros_like(vector)
            vector = diffusion.reverse_step(vector, times, generator.denoiser(vector, times), noise)
            completed = diffusion.config.steps-index
            if completed % 25 == 0 or completed in {diffusion.config.steps-1, diffusion.config.steps}:
                snapshots.append((completed, vector[0].clone()))
    return vector[0], snapshots


def load_validation_scores(path):
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        if "test_accuracy" in reader.fieldnames:
            raise ValueError("Diagnosis must not read official test results")
        rows = [row for row in reader if row["method"] in {"diffusion", "gaussian"}]
    scores = {}
    for row in rows:
        if row["status"] != "ok":
            raise ValueError("Expected the previously verified candidates")
        identity = (row["method"], int(row["seed"]))
        if identity in scores:
            raise ValueError("Duplicate candidate identity")
        scores[identity] = float(row["validation_accuracy"])
    if set(scores) != {(method, seed) for method in ("diffusion", "gaussian") for seed in range(40001, 40101)}:
        raise ValueError("Require exactly the existing locked candidate seeds")
    return scores


def correlation(x, y):
    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    x, y = x-x.mean(), y-y.mean()
    denominator = x.norm()*y.norm()
    return (x@y/denominator).item() if denominator > 0 else None


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_report(output, rows, traces, pca):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    cumulative = pca["cumulative_variance"].tolist()[:12]
    axes[0, 0].plot(range(1, len(cumulative)+1), cumulative, marker="o")
    axes[0, 0].axhline(.99, color="gray", linestyle="--")
    axes[0, 0].set(xlabel="Training principal components", ylabel="Cumulative variance fraction", title="Training latent dimension")
    for method in ("train", "validation", "diffusion", "gaussian"):
        group = [r for r in rows if r["method"] == method]
        for axis, key in ((axes[0, 1], "norm"), (axes[1, 0], "pca_residual_rms"), (axes[1, 1], "nearest_training_latent_rms")):
            axis.scatter([r[key] for r in group], [r["validation_accuracy"]*100 for r in group], alpha=.55, s=15, label=method)
    axes[0, 1].set(xlabel="Standardized latent norm", ylabel="Validation accuracy (%)", title="Norm and quality")
    axes[1, 0].set(xlabel="Training-PCA residual RMS", ylabel="Validation accuracy (%)", title="Deviation from training subspace")
    axes[1, 1].set(xlabel="Nearest-training-latent RMS", ylabel="Validation accuracy (%)", title="Training-latent proximity")
    axes[0, 1].legend()
    fig.savefig(output / "latent_diagnostics.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for seed in range(40001, 40006):
        group = [r for r in traces if r["seed"] == seed]
        for axis, key in zip(axes, ("norm", "pca_residual_rms")):
            axis.plot([r["completed_steps"] for r in group], [r[key] for r in group], label=str(seed))
    axes[0].set(xlabel="Completed reverse steps", ylabel="Latent norm", title="Five unchanged reverse trajectories")
    axes[1].set(xlabel="Completed reverse steps", ylabel="Training-PCA residual RMS", title="Noise-to-latent subspace deviation")
    axes[0].legend()
    fig.savefig(output / "reverse_trajectories.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="artifacts/diffusion-pilot/generator.pt")
    parser.add_argument("--latents", default="artifacts/autoencoder-pilot/latents.pt")
    parser.add_argument("--validation", default="artifacts/locked-evaluation/validation_candidates.csv")
    parser.add_argument("--protocol", default="artifacts/locked-evaluation/protocol.json")
    parser.add_argument("--output", default="artifacts/latent-diagnosis")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a fresh diagnosis directory")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(Path(args.protocol).read_text())
    for name, path in (("generator", args.bundle), ("latents", args.latents)):
        if file_hash(path) != protocol["artifacts"][name]["sha256"]:
            raise ValueError("Diagnosis artifacts differ from locked experiment")
    data = torch.load(args.latents, weights_only=True)
    training = training_latents(data)
    scores = load_validation_scores(args.validation)
    generator = Generator(torch.load(args.bundle, weights_only=True))
    pca = fit_pca(training)
    output.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started+900
    write_exclusive(output / "run.json", {"bundle_sha256": file_hash(args.bundle), "latents_sha256": file_hash(args.latents),
        "validation_sha256": file_hash(args.validation), "protocol_sha256": protocol["protocol_sha256"],
        "pca_fit_rows": data["training_rows"], "retained_training_variance": .99, "max_seconds": 900,
        "scope": "validation only, no dataset loading or optimization"})
    rows, traces = [], []
    for index, record in enumerate(data["records"]):
        rows.append({"method": record["split"], "seed": record["seed"], "source_step": record["step"],
                     "validation_accuracy": record["validation_accuracy"],
                     **describe_latent(data["standardized_latents"][index], training, pca,
                                      data["training_rows"].index(index) if record["split"] == "train" else None)})
    for method in ("diffusion", "gaussian"):
        for seed in range(40001, 40101):
            if time.perf_counter() >= deadline:
                write_csv(output / "latent_metrics.csv", rows)
                write_exclusive(output / "partial.json", {"complete": False, "rows": len(rows)})
                return
            if method == "diffusion":
                if seed <= 40005:
                    vector, snapshots = trace_sampling(generator, seed)
                    ordinary = generator.diffusion.sample(generator.denoiser, seed)[0]
                    torch.testing.assert_close(vector, ordinary, rtol=0, atol=0)
                    for step, value in snapshots:
                        traces.append({"seed": seed, "completed_steps": step,
                                       **describe_latent(value, training, pca)})
                else:
                    vector = generator.diffusion.sample(generator.denoiser, seed)[0]
            else:
                vector = torch.randn(128, generator=torch.Generator().manual_seed(seed))
            rows.append({"method": method, "seed": seed, "source_step": None,
                         "validation_accuracy": scores[(method, seed)], **describe_latent(vector, training, pca)})
            write_csv(output / "latent_metrics.csv", rows)
    write_csv(output / "trajectories.csv", traces)
    summary = {}
    for method in ("train", "validation", "diffusion", "gaussian"):
        group = [r for r in rows if r["method"] == method]
        summary[method] = {"count": len(group)}
        for key in ("norm", "max_abs_coordinate", "pca_residual_rms", "nearest_training_latent_rms"):
            values = [r[key] for r in group]
            summary[method][key] = {"median": statistics.median(values), "min": min(values), "max": max(values),
                "pearson_with_validation_accuracy": correlation(values, [r["validation_accuracy"] for r in group])}
    result = {"complete": True, "pca_rank_99_percent": pca["rank"],
              "first_component_variance": pca["cumulative_variance"][0].item(),
              "groups": summary, "elapsed_seconds_before_plot": time.perf_counter()-started,
              "interpretation": "exploratory correlations; no causal identification"}
    torch.save(pca, output / "training_pca.pt")
    write_exclusive(output / "results.json", result)
    plot_report(output, rows, traces, pca)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
