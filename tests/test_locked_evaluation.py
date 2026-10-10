import json
import platform

import pytest
import torch

from p_diff.evaluation_protocol import (
    file_hash, protocol_hash, require_selections, seal_selections, select_candidates,
    validate_protocol, write_exclusive,
)
from p_diff.locked_evaluation import load_official_test, proximity


def test_lock_detects_changed_protocol_and_changed_artifacts(tmp_path):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"original")
    protocol = {"candidate_seeds": list(range(40001, 40101)), "random_seeds": list(range(50001, 50006)),
        "counts": {"diffusion": 100, "gaussian": 100, "latent_mean": 1, "weight_average": 1, "random": 5},
        "source_counts": {"train": 160, "validation": 20, "held_out": 20},
        "training_allowed": False, "max_seconds": 1800,
        "artifacts": {"weights": {"path": str(artifact), "sha256": file_hash(artifact)}}, "code": {},
        "environment": {"torch": str(torch.__version__), "python": platform.python_version()}}
    protocol["protocol_sha256"] = protocol_hash(protocol)
    validate_protocol(protocol)
    protocol["training_allowed"] = True
    with pytest.raises(ValueError, match="contents changed"):
        validate_protocol(protocol)
    protocol["training_allowed"] = False
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_protocol(protocol)


def test_validation_selection_is_sealed_and_test_fields_are_forbidden(tmp_path):
    protocol = {"counts": {"diffusion": 2, "gaussian": 2}, "candidate_seeds": [1, 2],
                "random_seeds": [], "protocol_sha256": "fixed"}
    records = []
    for method in ("diffusion", "gaussian"):
        for seed in (1, 2):
            path = tmp_path / f"{method}-{seed}.pt"
            path.write_bytes(f"{method}-{seed}".encode())
            records.append({"method": method, "seed": seed, "status": "ok", "file": path.name,
                            "model_sha256": file_hash(path), "validation_accuracy": .94})
    with pytest.raises(ValueError, match="counts incomplete"):
        seal_selections(tmp_path / "selections.json", protocol, records[:-1])
    seal_selections(tmp_path / "selections.json", protocol, records)
    result = require_selections(tmp_path / "selections.json", protocol, tmp_path)
    assert result["selected"]["diffusion"]["seed"] == 1
    assert result["selected"]["gaussian"]["seed"] == 1
    with pytest.raises(FileExistsError):
        seal_selections(tmp_path / "selections.json", protocol, records)
    records[0]["test_accuracy"] = .01
    with pytest.raises(ValueError, match="before test results"):
        select_candidates(records)
    (tmp_path / result["selected"]["diffusion"]["file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="classifier changed"):
        require_selections(tmp_path / "selections.json", protocol, tmp_path)


def test_missing_selection_prevents_any_test_dataset_construction(tmp_path, monkeypatch):
    import torchvision.datasets
    def forbidden(*args, **kwargs):
        pytest.fail("Test dataset was accessed before selections existed")
    monkeypatch.setattr(torchvision.datasets, "MNIST", forbidden)
    with pytest.raises(FileNotFoundError):
        load_official_test("unavailable-data", tmp_path / "missing.json", {}, tmp_path)


def test_source_proximity_excludes_self_reference():
    bundle = {"normalization_mean": torch.zeros(2), "normalization_std": torch.ones(2)}
    sources = torch.tensor([[0., 0.], [1., 1.]])
    predictions = torch.tensor([[0, 0], [1, 1]])
    result = proximity(sources[0], predictions[0], sources, predictions, bundle, exclude_row=0)
    assert result["nearest_training_row"] == 1
    assert result["nearest_training_normalized_rms"] == 1
    assert result["minimum_source_test_disagreement"] == 1


def test_exclusive_protocol_write_cannot_overwrite_previous_lock(tmp_path):
    path = tmp_path / "protocol.json"
    write_exclusive(path, {"locked": True})
    with pytest.raises(FileExistsError):
        write_exclusive(path, {"locked": False})
    assert json.loads(path.read_text()) == {"locked": True}
