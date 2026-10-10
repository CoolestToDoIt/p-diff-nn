"""Run a frozen, validation-selected MNIST test evaluation without training."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

from .classifier import Classifier
from .data import mnist_training_data
from .evaluation_protocol import file_hash, require_selections, seal_selections, tensor_hash, validate_protocol, write_exclusive
from .export_latents import load_autoencoder
from .generate_models import Generator
from .metrics import classifier_predictions


class BudgetExceeded(Exception):
    pass


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cache_images(dataset):
    batches = list(DataLoader(dataset, batch_size=512))
    return torch.cat([x for x, _ in batches]), torch.cat([y for _, y in batches])


def load_official_test(data_root, selection_path, protocol, output):
    # This gate must complete before even constructing the official test dataset.
    require_selections(selection_path, protocol, output)
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    dataset = MNIST(data_root, train=False, download=False, transform=ToTensor())
    if len(dataset) != 10000:
        raise ValueError("Expected the official 10,000-image MNIST test set")
    return cache_images(dataset)


def accuracy_summary(values):
    if not values:
        return {}
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "std": statistics.pstdev(values), "worst": min(values), "best": max(values)}


@torch.no_grad()
def proximity(vector, predictions, source_vectors, source_predictions, bundle, exclude_row=None):
    normalized = (vector - bundle["normalization_mean"]) / bundle["normalization_std"]
    distances = (source_vectors - normalized[None]).square().mean(1).sqrt()
    disagreements = (source_predictions != predictions[None]).float().mean(1)
    if exclude_row is not None:
        distances[exclude_row] = float("inf")
        disagreements[exclude_row] = float("inf")
    nearest = distances.argmin().item()
    finite_disagreement = disagreements[torch.isfinite(disagreements)]
    return {"nearest_training_row": nearest, "nearest_training_normalized_rms": distances[nearest].item(),
            "nearest_weight_source_test_disagreement": disagreements[nearest].item(),
            "minimum_source_test_disagreement": finite_disagreement.min().item(),
            "mean_source_test_disagreement": finite_disagreement.mean().item()}


def pairwise_disagreement(predictions):
    if len(predictions) < 2:
        return None
    return statistics.mean((predictions[i] != predictions[j]).float().mean().item()
                           for i in range(len(predictions)) for j in range(i+1, len(predictions)))


def plot_report(output, candidates, source_median, reconstructions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ("diffusion", "gaussian", "latent_mean", "weight_average", "random")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for i, method in enumerate(methods):
        rows = [r for r in candidates if r["method"] == method and r["status"] == "ok"]
        axes[0].scatter([i]*len(rows), [100*r["test_accuracy"] for r in rows], alpha=.45, s=18)
        if method != "random":
            axes[1].scatter([r["nearest_training_normalized_rms"] for r in rows],
                            [100*r["nearest_weight_source_test_disagreement"] for r in rows], label=method, alpha=.55, s=18)
    axes[0].axhline(source_median*100, color="black", linestyle="--")
    axes[0].axhspan((source_median-.02)*100, (source_median+.02)*100, color="green", alpha=.1)
    axes[0].set(xticks=range(5), xticklabels=["Diffusion", "Gaussian", "Latent mean", "Weight average", "Random"],
                ylabel="Official test accuracy (%)", title="Every locked candidate")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].set(xlabel="Nearest-training-source normalized weight RMS", ylabel="Test prediction disagreement (%)",
                title="Source proximity")
    axes[1].legend()
    fig.savefig(output / "test_comparison.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    axis.plot([r["step"] for r in reconstructions], [100*r["original_test_accuracy"] for r in reconstructions], label="Original")
    axis.plot([r["step"] for r in reconstructions], [100*r["reconstructed_test_accuracy"] for r in reconstructions], label="Reconstructed")
    axis.set(xlabel="Held-out branch training step", ylabel="Official test accuracy (%)", title="Frozen autoencoder on held-out branch")
    axis.legend()
    fig.savefig(output / "held_out_reconstruction.png", dpi=160)
    plt.close(fig)


def run(protocol, output, data_root):
    validate_protocol(protocol)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh evaluation directory; partial runs are retained")
    output.mkdir(parents=True)
    (output / "models").mkdir()
    write_exclusive(output / "protocol.json", protocol)
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    deadline = started + protocol["max_seconds"]
    candidates, sources, reconstructions = [], [], []
    stage = "load_artifacts"
    timings = {}

    def budget():
        if time.perf_counter() >= deadline:
            raise BudgetExceeded(stage)

    def artifact(name):
        return torch.load(protocol["artifacts"][name]["path"], weights_only=True)

    try:
        begin = time.perf_counter()
        generator = Generator(artifact("generator"))
        timings["bundle_load_seconds"] = time.perf_counter()-begin
        for key, digest in protocol["statistics"].items():
            if tensor_hash(generator.bundle[key]) != digest:
                raise ValueError("Normalization statistics changed")
        data = artifact("parameters")
        if Counter(r["split"] for r in data["records"]) != protocol["source_counts"]:
            raise ValueError("Source counts differ from frozen protocol")
        if data["manifest"] != generator.codec.manifest():
            raise ValueError("Source/generator architecture mismatch")
        training_rows = [i for i, r in enumerate(data["records"]) if r["split"] == "train"]
        if data["normalization_fit_rows"].tolist() != training_rows:
            raise ValueError("Normalization was not fit exclusively on training sources")
        for key in ("normalization_mean", "normalization_std"):
            torch.testing.assert_close(data[key], generator.bundle[key], rtol=0, atol=0)
        _, validation = mnist_training_data(data_root, artifact("splits"))
        val_images, val_labels = cache_images(validation)
        stage = "generate_and_validate"
        requests = [(method, seed) for method in ("diffusion", "gaussian") for seed in protocol["candidate_seeds"]]
        requests += [("latent_mean", None), ("weight_average", None)]
        requests += [("random", seed) for seed in protocol["random_seeds"]]
        for method, seed in requests:
            budget()
            begin = time.perf_counter()
            try:
                if method == "weight_average":
                    vector = data["vectors"][training_rows].mean(0)
                elif method == "random":
                    with torch.random.fork_rng():
                        torch.manual_seed(seed)
                        vector = generator.codec.flatten(Classifier())
                else:
                    vector = generator.generate_vector(seed or 0, method)
                model = generator.codec.restore(vector, Classifier()).eval()
                sampling_seconds = time.perf_counter()-begin
                begin = time.perf_counter()
                filename = f"models/{method}-{seed if seed is not None else 'single'}.pt"
                torch.save({"state_dict": model.state_dict(), "manifest": generator.codec.manifest(),
                            "method": method, "seed": seed, "protocol_sha256": protocol["protocol_sha256"]}, output / filename)
                export_seconds = time.perf_counter()-begin
                begin = time.perf_counter()
                pred = classifier_predictions(model, val_images)
                record = {"method": method, "seed": seed, "status": "ok", "file": filename,
                          "model_sha256": file_hash(output / filename),
                          "validation_accuracy": (pred == val_labels).double().mean().item(),
                          "generation_seconds": sampling_seconds, "export_seconds": export_seconds,
                          "validation_seconds": time.perf_counter()-begin}
            except ValueError as error:
                record = {"method": method, "seed": seed, "status": "failed", "error": str(error)}
            candidates.append(record)
            write_csv(output / "validation_candidates.csv", candidates)
        budget()
        seal_selections(output / "selections.json", protocol, candidates)
        selections = require_selections(output / "selections.json", protocol, output)
        timings["validation_stage_seconds"] = time.perf_counter()-started
        # Revalidate every locked input and code hash at the exact test-access boundary.
        validate_protocol(protocol)
        write_exclusive(output / "test_access.json", {"protocol_sha256": protocol["protocol_sha256"],
            "selections_sha256": file_hash(output / "selections.json"),
            "validation_stage_complete": True, "seconds_since_start": time.perf_counter()-started})
        stage = "official_test_sources"
        begin = time.perf_counter()
        test_images, test_labels = load_official_test(data_root, output / "selections.json", protocol, output)
        timings["test_data_load_seconds"] = time.perf_counter()-begin
        del val_images, val_labels
        begin = time.perf_counter()
        source_predictions = []
        for index, provenance in enumerate(data["records"]):
            budget()
            model = generator.codec.restore(data["vectors"][index], Classifier()).eval()
            pred = classifier_predictions(model, test_images)
            source_predictions.append(pred)
            sources.append({"row": index, "branch_id": provenance["branch_id"], "split": provenance["split"],
                            "seed": provenance["seed"], "step": provenance["step"],
                            "test_accuracy": (pred == test_labels).double().mean().item()})
            write_csv(output / "sources.csv", sources)
        source_predictions = torch.stack(source_predictions)
        train_predictions = source_predictions[training_rows]
        train_vectors = data["normalized_vectors"][training_rows]
        training_local_rows = {global_row: local_row for local_row, global_row in enumerate(training_rows)}
        for row in sources:
            budget()
            row.update(proximity(data["vectors"][row["row"]], source_predictions[row["row"]],
                                 train_vectors, train_predictions, generator.bundle,
                                 training_local_rows.get(row["row"])))
        write_csv(output / "sources.csv", sources)
        timings["source_test_and_proximity_seconds"] = time.perf_counter()-begin
        source_median = statistics.median(sources[i]["test_accuracy"] for i in training_rows)
        stage = "official_test_candidates"
        begin = time.perf_counter()
        candidate_predictions = {}
        for row in candidates:
            budget()
            if row["status"] != "ok":
                continue
            if file_hash(output / row["file"]) != row["model_sha256"]:
                raise ValueError("Saved candidate changed after validation")
            checkpoint = torch.load(output / row["file"], weights_only=True)
            model = Classifier().eval()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            start_test = time.perf_counter()
            pred = classifier_predictions(model, test_images)
            row["test_accuracy"] = (pred == test_labels).double().mean().item()
            row.update(proximity(generator.codec.flatten(model), pred, train_vectors, train_predictions, generator.bundle))
            row["test_and_proximity_seconds"] = time.perf_counter()-start_test
            candidate_predictions.setdefault(row["method"], []).append(pred)
            write_csv(output / "candidates.csv", candidates)
        timings["candidate_test_and_proximity_seconds"] = time.perf_counter()-begin
        stage = "held_out_reconstruction"
        begin = time.perf_counter()
        autoencoder = load_autoencoder(artifact("autoencoder"))
        for index, provenance in enumerate(data["records"]):
            if provenance["split"] != "held_out":
                continue
            budget()
            with torch.no_grad():
                normalized = autoencoder(data["normalized_vectors"][index:index+1])[0]
                vector = normalized*data["normalization_std"]+data["normalization_mean"]
            model = generator.codec.restore(vector, Classifier()).eval()
            pred = classifier_predictions(model, test_images)
            value = (pred == test_labels).double().mean().item()
            reconstructions.append({"row": index, "step": provenance["step"],
                "original_test_accuracy": sources[index]["test_accuracy"], "reconstructed_test_accuracy": value,
                "accuracy_loss": sources[index]["test_accuracy"]-value})
            write_csv(output / "held_out_reconstruction.csv", reconstructions)
        timings["held_out_reconstruction_seconds"] = time.perf_counter()-begin
        stage = "report"
        methods = {}
        for method, expected in protocol["counts"].items():
            rows = [r for r in candidates if r["method"] == method]
            good = [r for r in rows if r["status"] == "ok"]
            values = [r["test_accuracy"] for r in good]
            fraction = sum(abs(value-source_median) <= protocol["target"]["accuracy_tolerance"] for value in values)/expected
            metrics = {**accuracy_summary(values), "requested_count": expected, "failures": expected-len(good),
                       "fraction_within_two_points": fraction,
                       "quality_target_passed": bool(values) and abs(statistics.median(values)-source_median) <= .02 and fraction >= .8,
                       "mean_candidate_test_disagreement": pairwise_disagreement(candidate_predictions.get(method, []))}
            if method in selections["selected"]:
                chosen = selections["selected"][method]
                record = next(r for r in good if r["seed"] == chosen["seed"])
                metrics["selected_seed"] = chosen["seed"]
                metrics["selected_validation_accuracy"] = chosen["validation_accuracy"]
                metrics["selected_test_accuracy"] = record["test_accuracy"]
            if good:
                for key in ("generation_seconds", "validation_seconds", "export_seconds", "test_and_proximity_seconds",
                            "nearest_training_normalized_rms", "nearest_weight_source_test_disagreement"):
                    metrics[f"mean_{key}"] = statistics.mean(r[key] for r in good)
            methods[method] = metrics
        source_summary = {}
        for split in protocol["source_counts"]:
            rows = [r for r in sources if r["split"] == split]
            source_summary[split] = {**accuracy_summary([r["test_accuracy"] for r in rows]),
                "mean_nearest_other_training_normalized_rms": statistics.mean(r["nearest_training_normalized_rms"] for r in rows),
                "mean_nearest_other_weight_source_test_disagreement": statistics.mean(r["nearest_weight_source_test_disagreement"] for r in rows)}
        original = statistics.median(r["original_test_accuracy"] for r in reconstructions)
        rebuilt = statistics.median(r["reconstructed_test_accuracy"] for r in reconstructions)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {"complete": True, "protocol_sha256": protocol["protocol_sha256"],
                  "source_training_median_test_accuracy": source_median, "methods": methods, "sources": source_summary,
                  "held_out_reconstruction": {"count": len(reconstructions), "median_original": original,
                    "median_reconstructed": rebuilt, "difference_of_medians": original-rebuilt,
                    "loss_distribution": accuracy_summary([r["accuracy_loss"] for r in reconstructions])},
                  "timings": timings, "elapsed_seconds_before_plot": time.perf_counter()-started,
                  "process_peak_rss_bytes": peak if sys.platform == "darwin" else peak*1024,
                  "training_performed": False}
        write_exclusive(output / "results.json", result)
        budget()
        plot_report(output, candidates, source_median, reconstructions)
        write_exclusive(output / "completion.json", {"complete": True, "elapsed_seconds": time.perf_counter()-started,
                                                      "results_sha256": file_hash(output / "results.json")})
        print(json.dumps(result, indent=2), flush=True)
    except BudgetExceeded:
        write_csv(output / "candidates_partial.csv", candidates)
        write_exclusive(output / "partial.json", {"complete": False, "stage": stage,
            "elapsed_seconds": time.perf_counter()-started, "candidate_count": len(candidates),
            "source_count": len(sources), "reconstruction_count": len(reconstructions),
            "protocol_sha256": protocol["protocol_sha256"]})
        print(f"Budget exhausted during {stage}; partial artifacts retained", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="artifacts/locked-evaluation-protocol.json")
    parser.add_argument("--output", default="artifacts/locked-evaluation")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    run(json.loads(Path(args.protocol).read_text()), args.output, args.data_root)
