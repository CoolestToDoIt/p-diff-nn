"""Run the bounded reconstruction pilot, with a single larger-latent retry."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from .autoencoder import AutoencoderConfig, ParameterAutoencoder, balanced_loss, block_errors
from .classifier import Classifier
from .data import mnist_training_data
from .parameters import BlockNormalizer, ParameterCodec
from .train_baseline import accuracy


def load_inputs(path):
    data = torch.load(path, weights_only=True)
    if any(r["split"] not in {"train", "validation"} for r in data["records"]):
        raise ValueError("Prepare inputs without held-out rows before training")
    codec = ParameterCodec(Classifier())
    if data["manifest"] != codec.manifest():
        raise ValueError("Source manifest mismatch")
    rows = {split: [i for i, r in enumerate(data["records"]) if r["split"] == split]
            for split in ("train", "validation")}
    if not all(rows.values()):
        raise ValueError("Need both training and validation branches")
    branch_splits = {}
    for record in data["records"]:
        branch = record["branch_id"]
        if branch in branch_splits and branch_splits[branch] != record["split"]:
            raise ValueError("Branch crosses dataset splits")
        branch_splits[branch] = record["split"]
    fit = BlockNormalizer.fit(data["vectors"][rows["train"]], codec)
    torch.testing.assert_close(fit.mean, data["normalization_mean"], rtol=0, atol=0)
    torch.testing.assert_close(fit.std, data["normalization_std"], rtol=0, atol=0)
    torch.testing.assert_close(fit.normalize(data["vectors"]), data["normalized_vectors"])
    return data, rows, codec


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output, history, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot([h["step"] for h in history], [h["validation_loss"] for h in history])
    axes[0].set(xlabel="Optimizer updates", ylabel="Balanced normalized MSE", yscale="log",
                title="Validation reconstruction error")
    axes[1].plot([r["step"] for r in records], [100*r["original_accuracy"] for r in records], label="Source")
    axes[1].plot([r["step"] for r in records], [100*r["reconstructed_accuracy"] for r in records], label="Reconstructed")
    axes[1].set(xlabel="Source branch step", ylabel="Validation accuracy (%)",
                title="Validation branch classifiers")
    axes[1].legend()
    fig.savefig(output / "reconstruction.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="artifacts/source-full/reconstruction-inputs.pt")
    parser.add_argument("--splits", default="artifacts/source-full/splits.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--config", default="configs/autoencoder_pilot.yaml")
    parser.add_argument("--output", default="artifacts/autoencoder-pilot")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if (config["latent_sizes"] != [128, 256] or not 1 <= config["max_updates"] <= 1000
            or config["batch_size"] != 16 or config["learning_rate"] != .001
            or config["validation_interval"] != 100 or not 0 < config["max_seconds"] <= 1800):
        raise ValueError("Settings exceed or differ from the authorized pilot")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data, rows, codec = load_inputs(args.inputs)
    split_hash = hashlib.sha256(Path(args.splits).read_bytes()).hexdigest()
    if any(r["split_sha256"] != split_hash for r in data["records"]):
        raise ValueError("Image split identity mismatch")
    _, validation = mnist_training_data(args.data_root, torch.load(args.splits, weights_only=True))
    # Cache only the fixed validation images to avoid repeated image conversions.
    image_batches, label_batches = [], []
    for images, labels in DataLoader(validation, batch_size=512):
        image_batches.append(images)
        label_batches.append(labels)
    image_loader = DataLoader(TensorDataset(torch.cat(image_batches), torch.cat(label_batches)), batch_size=512)
    train_vectors = data["normalized_vectors"][rows["train"]]
    val_vectors = data["normalized_vectors"][rows["validation"]]
    normalizer = BlockNormalizer(data["normalization_mean"], data["normalization_std"])
    output.mkdir(parents=True)
    input_hash = hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest()
    started = time.perf_counter()
    deadline = started + config["max_seconds"]
    attempts = []
    for latent_size in config["latent_sizes"]:
        torch.manual_seed(config["seed"])
        generator = torch.Generator().manual_seed(config["seed"])
        architecture = AutoencoderConfig(latent_size=latent_size)
        model = ParameterAutoencoder(architecture)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        directory = output / str(latent_size)
        directory.mkdir()
        history = []
        best_loss = float("inf")
        attempt_start = time.perf_counter()
        completed = 0
        for step in range(1, config["max_updates"] + 1):
            if time.perf_counter() >= deadline:
                break
            model.train()
            batch = train_vectors[torch.randint(len(train_vectors), (config["batch_size"],), generator=generator)]
            optimizer.zero_grad(set_to_none=True)
            loss = balanced_loss(model(batch), batch, codec.specs)
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
            completed = step
            if step == 5:
                smoke_seconds = time.perf_counter() - attempt_start
                print(json.dumps({"latent_size": latent_size, "five_update_seconds": smoke_seconds,
                                  "estimated_1000_update_seconds": smoke_seconds * 200}), flush=True)
            if step % config["validation_interval"] and step != config["max_updates"]:
                continue
            model.eval()
            with torch.no_grad():
                val_loss = balanced_loss(model(val_vectors), val_vectors, codec.specs).item()
            if not torch.isfinite(torch.tensor(val_loss)):
                break
            history.append({"step": step, "training_loss": loss.item(), "validation_loss": val_loss,
                            "elapsed_seconds": time.perf_counter() - attempt_start})
            (directory / "history.json").write_text(json.dumps(history, indent=2))
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"state_dict": model.state_dict(), "architecture": architecture.manifest(),
                            "input_sha256": input_hash, "source_sha256": data["source_sha256"],
                            "manifest": codec.manifest(), "normalization_mean": normalizer.mean,
                            "normalization_std": normalizer.std, "config": config,
                            "best_step": step, "validation_loss": val_loss}, directory / "best.pt")
            print(json.dumps({"latent_size": latent_size, **history[-1]}), flush=True)
        if not (directory / "best.pt").exists():
            torch.save({"state_dict": model.state_dict(), "architecture": architecture.manifest(),
                        "step": completed}, directory / "partial.pt")
            attempts.append({"latent_size": latent_size, "status": "stopped_before_validation", "gate_passed": False})
            break
        best = torch.load(directory / "best.pt", weights_only=True)
        model.load_state_dict(best["state_dict"])
        model.eval().requires_grad_(False)
        with torch.no_grad():
            reconstructed_normalized = model(val_vectors)
            reconstructed = normalizer.inverse(reconstructed_normalized)
            errors = block_errors(reconstructed_normalized, val_vectors, codec.specs).tolist()
        records = []
        for position, index in enumerate(rows["validation"]):
            if time.perf_counter() >= deadline:
                break
            original = codec.restore(data["vectors"][index], Classifier()).eval()
            restored = codec.restore(reconstructed[position], Classifier()).eval()
            original_accuracy = accuracy(original, image_loader)
            rebuilt_accuracy = accuracy(restored, image_loader)
            if abs(original_accuracy - data["records"][index]["validation_accuracy"]) > 1e-7:
                raise ValueError("Source validation accuracy no longer matches provenance")
            records.append({"branch_id": data["records"][index]["branch_id"],
                            "step": data["records"][index]["step"], "original_accuracy": original_accuracy,
                            "reconstructed_accuracy": rebuilt_accuracy,
                            "accuracy_loss": original_accuracy - rebuilt_accuracy})
        if records:
            write_csv(directory / "reconstruction.csv", records)
            plot_results(directory, history, records)
        complete = len(records) == len(rows["validation"])
        median_loss = (statistics.median(r["original_accuracy"] for r in records) -
                       statistics.median(r["reconstructed_accuracy"] for r in records)) if records else None
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {"latent_size": latent_size, "updates_completed": completed, "best_step": best["best_step"],
                  "validation_loss": best_loss, "evaluated_classifiers": len(records),
                  "gate_passed": complete and median_loss <= .01, "median_accuracy_loss": median_loss,
                  "median_original_accuracy": statistics.median(r["original_accuracy"] for r in records) if records else None,
                  "median_reconstructed_accuracy": statistics.median(r["reconstructed_accuracy"] for r in records) if records else None,
                  "per_checkpoint_accuracy_losses": [r["accuracy_loss"] for r in records],
                  "block_mse": dict(zip([s.name for s in codec.specs], errors)),
                  "elapsed_seconds": time.perf_counter() - attempt_start,
                  "process_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024}
        (directory / "results.json").write_text(json.dumps(result, indent=2))
        best["gate_passed"] = result["gate_passed"]
        torch.save(best, directory / "best.pt")
        attempts.append(result)
        (output / "results.json").write_text(json.dumps({"attempts": attempts,
            "elapsed_seconds": time.perf_counter() - started}, indent=2))
        print(json.dumps(result), flush=True)
        if result["gate_passed"] or time.perf_counter() >= deadline:
            break
    (output / "results.json").write_text(json.dumps({"attempts": attempts,
        "elapsed_seconds": time.perf_counter() - started}, indent=2))


if __name__ == "__main__":
    main()
