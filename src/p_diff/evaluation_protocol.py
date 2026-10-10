"""Hash-locked experiment protocol and validation-only candidate selection."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import torch


ARTIFACTS = {
    "generator": "artifacts/diffusion-pilot/generator.pt",
    "parameters": "artifacts/source-full/parameters.pt",
    "autoencoder": "artifacts/autoencoder-pilot/128/best.pt",
    "latents": "artifacts/autoencoder-pilot/latents.pt",
    "splits": "artifacts/source-full/splits.pt",
}
CODE_FILES = [f"src/p_diff/{name}.py" for name in (
    "evaluation_protocol", "locked_evaluation", "diffusion", "generate_models", "autoencoder",
    "export_latents", "train_autoencoder", "metrics", "data", "parameters", "classifier")]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def protocol_hash(protocol):
    body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_exclusive(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def create_protocol():
    bundle = torch.load(ARTIFACTS["generator"], weights_only=True)
    if bundle["metadata"]["step"] != 2000 or bundle["diffusion_config"] != {
        "steps": 200, "beta_start": .0005, "beta_end": .1}:
        raise ValueError("Expected the frozen step-2000 pilot generator")
    protocol = {
        "version": 1, "base_seed": 42, "threads": 1, "max_seconds": 1800,
        "artifacts": {key: {"path": path, "sha256": file_hash(path)} for key, path in ARTIFACTS.items()},
        "code": {path: file_hash(path) for path in CODE_FILES},
        "statistics": {key: tensor_hash(bundle[key]) for key in (
            "latent_mean", "latent_std", "normalization_mean", "normalization_std")},
        "candidate_seeds": list(range(40001, 40101)), "random_seeds": list(range(50001, 50006)),
        "counts": {"diffusion": 100, "gaussian": 100, "latent_mean": 1, "weight_average": 1, "random": 5},
        "source_counts": {"train": 160, "validation": 20, "held_out": 20},
        "selection": "highest validation accuracy; lowest seed on ties; write before test access",
        "reference": "median official-test accuracy of the 160 training-source checkpoints",
        "target": {"accuracy_tolerance": .02, "minimum_fraction": .8},
        "metrics": {"accuracy_std": "population", "distance": "nearest-training-source normalized full-vector RMS",
                    "disagreement": "fraction of differing official-test predictions",
                    "self_comparisons": "exclude from source-to-other-source references"},
        "preprocessing": "MNIST ToTensor, [0,1], no augmentation; official train/validation/test split",
        "diffusion_config": bundle["diffusion_config"], "latent_size": 128,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "platform": platform.system(), "machine": platform.machine()},
        "training_allowed": False,
    }
    protocol["protocol_sha256"] = protocol_hash(protocol)
    return protocol


def validate_protocol(protocol):
    if protocol.get("protocol_sha256") != protocol_hash(protocol):
        raise ValueError("Protocol contents changed after locking")
    if (protocol["candidate_seeds"] != list(range(40001, 40101)) or protocol["random_seeds"] != list(range(50001, 50006))
            or protocol["counts"] != {"diffusion": 100, "gaussian": 100, "latent_mean": 1, "weight_average": 1, "random": 5}
            or protocol["source_counts"] != {"train": 160, "validation": 20, "held_out": 20}
            or protocol["training_allowed"] or protocol["max_seconds"] != 1800):
        raise ValueError("Protocol differs from the authorized fixed scope")
    for artifact in protocol["artifacts"].values():
        if file_hash(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Locked artifact changed: {artifact['path']}")
    for path, digest in protocol["code"].items():
        if file_hash(path) != digest:
            raise ValueError(f"Locked evaluation code changed: {path}")
    if protocol["environment"]["torch"] != str(torch.__version__) or protocol["environment"]["python"] != platform.python_version():
        raise ValueError("Runtime differs from locked Python/PyTorch versions")


def select_candidates(records):
    if any("test_accuracy" in row for row in records):
        raise ValueError("Selections must be sealed before test results exist")
    selected = {}
    for method in ("diffusion", "gaussian"):
        successful = [row for row in records if row["method"] == method and row["status"] == "ok"]
        if not successful:
            raise ValueError(f"No usable validation candidate for {method}")
        row = max(successful, key=lambda r: (r["validation_accuracy"], -r["seed"]))
        selected[method] = {key: row[key] for key in ("method", "seed", "file", "model_sha256", "validation_accuracy")}
    return selected


def seal_selections(path, protocol, records):
    if Counter(row["method"] for row in records) != protocol["counts"]:
        raise ValueError("Candidate counts incomplete; test access is forbidden")
    for method in ("diffusion", "gaussian", "random"):
        expected = protocol["random_seeds"] if method == "random" else protocol["candidate_seeds"]
        if sorted(row["seed"] for row in records if row["method"] == method) != expected:
            raise ValueError("Candidate seed list differs from locked protocol")
    write_exclusive(path, {"protocol_sha256": protocol["protocol_sha256"], "selected": select_candidates(records)})


def require_selections(path, protocol, output):
    selections = json.loads(Path(path).read_text())
    if selections["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("Selections refer to a different protocol")
    for row in selections["selected"].values():
        if file_hash(Path(output) / row["file"]) != row["model_sha256"]:
            raise ValueError("Selected classifier changed before test access")
    return selections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/locked-evaluation-protocol.json")
    args = parser.parse_args()
    protocol = create_protocol()
    write_exclusive(args.output, protocol)
    print(f"Locked protocol {protocol['protocol_sha256']} at {args.output}; test data has not been accessed")
