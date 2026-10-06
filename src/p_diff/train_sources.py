"""Collect bounded shared-base checkpoint branches and export their vectors."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import torch
import yaml
from torch.utils.data import DataLoader

from .classifier import Classifier
from .data import mnist_training_data
from .parameters import BlockNormalizer, ParameterCodec


def export_dataset(vectors, records, codec):
    vectors = torch.stack(vectors)
    train_mask = torch.tensor([r["split"] == "train" for r in records])
    normalizer = BlockNormalizer.fit(vectors[train_mask], codec)
    normalized = normalizer.normalize(vectors)
    torch.testing.assert_close(normalizer.inverse(normalized), vectors)
    return {"vectors": vectors, "normalized_vectors": normalized,
            "normalization_mean": normalizer.mean, "normalization_std": normalizer.std,
            "records": records, "manifest": codec.manifest(),
            "normalization_fit_rows": torch.where(train_mask)[0]}


@torch.no_grad()
def predictions(model, loader):
    model.eval()
    return torch.cat([model(images).argmax(1) for images, _ in loader])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="artifacts/pilot-baseline")
    parser.add_argument("--config", default="configs/source_pilot.yaml")
    parser.add_argument("--output", default="artifacts/source-pilot")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if len(config["branch_seeds"]) != len(config["branch_splits"]):
        raise ValueError("Every branch needs a split")
    if not set(config["branch_splits"]) <= {"train", "validation", "held_out"}:
        raise ValueError("Invalid branch split")
    if "train" not in config["branch_splits"] or len(set(config["branch_seeds"])) != len(config["branch_seeds"]):
        raise ValueError("Need a training branch and distinct branch seeds")
    if min(config["epochs"], config["snapshots_per_branch"], config["batch_size"], config["learning_rate"]) <= 0:
        raise ValueError("Training settings must be positive")
    output, base = Path(args.output), Path(args.base)
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    checkpoint = torch.load(base / "baseline.pt", weights_only=True)
    if checkpoint["synthetic"]:
        raise ValueError("A synthetic baseline cannot seed the MNIST pilot")
    base_hash = hashlib.sha256((base / "baseline.pt").read_bytes()).hexdigest()
    split_hash = hashlib.sha256((base / "splits.pt").read_bytes()).hexdigest()
    splits = torch.load(base / "splits.pt", weights_only=True)
    baseline_metrics = json.loads((base / "metrics.json").read_text())
    remaining = config["max_total_training_seconds"] - baseline_metrics["elapsed_seconds"]
    if remaining <= 0:
        raise TimeoutError("Baseline consumed the total training budget")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train, validation = mnist_training_data(args.data_root, splits)
    validation_loader = DataLoader(validation, batch_size=config["batch_size"])
    labels_validation = torch.cat([y for _, y in validation_loader])
    codec = ParameterCodec(Classifier())
    output.mkdir(parents=True)
    torch.save(splits, output / "splits.pt")
    (output / "run.json").write_text(json.dumps({"config": config, "base_sha256": base_hash,
                                                 "split_sha256": split_hash}, indent=2))
    vectors, records, prediction_vectors = [], [], []
    branch_times = []
    started = time.perf_counter()
    stopped = False
    for branch, (seed, split) in enumerate(zip(config["branch_seeds"], config["branch_splits"])):
        torch.manual_seed(seed)
        model = Classifier()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        total_steps = len(loader) * config["epochs"]
        if config["snapshots_per_branch"] > total_steps:
            raise ValueError("More snapshots than training steps")
        steps = set(torch.linspace(1, total_steps, config["snapshots_per_branch"] + 1)[1:].round().int().tolist())
        branch_start = time.perf_counter()
        step = 0
        for epoch in range(config["epochs"]):
            for images, labels in loader:
                if time.perf_counter() - started >= remaining:
                    stopped = True
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()
                step += 1
                if step not in steps:
                    continue
                vector = codec.flatten(model)
                fresh = codec.restore(vector, Classifier()).eval()
                model.eval()
                torch.testing.assert_close(model(images), fresh(images), rtol=0, atol=0)
                # Reserve held-out behavioral metrics for locked evaluation.
                pred = None if split == "held_out" else predictions(fresh, validation_loader)
                record = {"branch_id": branch, "split": split, "seed": seed,
                          "base_seed": checkpoint["seed"], "base_sha256": base_hash,
                          "step": step, "epoch": epoch + 1,
                          "split_sha256": split_hash,
                          "validation_accuracy": None if pred is None else (pred == labels_validation).double().mean().item(),
                          "optimizer": "Adam", "learning_rate": config["learning_rate"],
                          "optimizer_defaults": {k: v for k, v in optimizer.defaults.items() if k != "lr"},
                          "batch_size": config["batch_size"],
                          "elapsed_seconds": time.perf_counter() - branch_start,
                          "file": f"branch-{branch}-step-{step}.pt"}
                torch.save({"state_dict": fresh.state_dict(), "provenance": record}, output / record["file"])
                records.append(record)
                vectors.append(vector)
                prediction_vectors.append(pred)
                (output / "provenance.json").write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)
            if stopped:
                break
        branch_times.append(time.perf_counter() - branch_start)
        if stopped:
            break
    if not vectors:
        raise TimeoutError("Budget exhausted before first snapshot; split metadata retained")
    dataset = export_dataset(vectors, records, codec)
    torch.save(dataset, output / "parameters.pt")
    (output / "manifest.json").write_text(json.dumps(codec.manifest(), indent=2))
    with (output / "inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    pairs = []
    for i, j in itertools.combinations(range(len(records)), 2):
        if records[i]["split"] == "held_out" or records[j]["split"] == "held_out":
            continue
        distance = (dataset["normalized_vectors"][i] - dataset["normalized_vectors"][j]).square().mean().sqrt().item()
        pairs.append({"i": i, "j": j, "same_branch": records[i]["branch_id"] == records[j]["branch_id"],
                      "normalized_weight_rms_distance": distance,
                      "validation_prediction_disagreement": (prediction_vectors[i] != prediction_vectors[j]).float().mean().item()})
    summary = {"config": config, "base_sha256": base_hash, "complete": not stopped,
               "baseline_seconds": baseline_metrics["elapsed_seconds"], "branch_seconds": branch_times,
               "pilot_seconds": time.perf_counter() - started,
               "snapshot_count": len(records), "pairs": pairs,
               "validation_accuracy": [r["validation_accuracy"] for r in records if r["split"] != "held_out"]}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
