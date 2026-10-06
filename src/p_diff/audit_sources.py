"""Verify source artifacts and summarize only training/validation branches."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

import torch

from .classifier import Classifier
from .parameters import BlockNormalizer, ParameterCodec


def describe(values):
    return {"min": min(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "max": max(values)}


def audit(root):
    root = Path(root)
    torch.set_num_threads(1)
    data = torch.load(root / "parameters.pt", weights_only=True)
    run = json.loads((root / "run.json").read_text())
    records = data["records"]
    codec = ParameterCodec(Classifier())
    assert data["manifest"] == codec.manifest()
    assert torch.isfinite(data["vectors"]).all()
    assert torch.unique(data["vectors"], dim=0).shape[0] == len(records)
    assert hashlib.sha256((root / "splits.pt").read_bytes()).hexdigest() == run["split_sha256"]
    splits = torch.load(root / "splits.pt", weights_only=True)
    assert len(splits["train"]) == 50000 and len(splits["validation"]) == 10000
    assert torch.cat([splits["train"], splits["validation"]]).unique().numel() == 60000
    train_rows = [i for i, r in enumerate(records) if r["split"] == "train"]
    assert data["normalization_fit_rows"].tolist() == train_rows
    fitted = BlockNormalizer.fit(data["vectors"][train_rows], codec)
    torch.testing.assert_close(fitted.mean, data["normalization_mean"], rtol=0, atol=0)
    torch.testing.assert_close(fitted.std, data["normalization_std"], rtol=0, atol=0)
    torch.testing.assert_close(fitted.inverse(data["normalized_vectors"]), data["vectors"])
    seen = set()
    for i, r in enumerate(records):
        assert r["split"] == run["config"]["branch_splits"][r["branch_id"]]
        assert r["seed"] == run["config"]["branch_seeds"][r["branch_id"]]
        assert r["base_sha256"] == run["base_sha256"] and r["split_sha256"] == run["split_sha256"]
        identity = (r["branch_id"], r["step"])
        assert identity not in seen
        seen.add(identity)
        checkpoint = torch.load(root / r["file"], weights_only=True)
        assert checkpoint["provenance"] == r
        model = Classifier()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        assert torch.equal(codec.flatten(model), data["vectors"][i])
        if r["split"] == "held_out":
            assert r["validation_accuracy"] is None
    metrics = json.loads((root / "metrics.json").read_text())
    for pair in metrics["pairs"]:
        assert records[pair["i"]]["split"] != "held_out"
        assert records[pair["j"]]["split"] != "held_out"
    accuracy = {split: describe([r["validation_accuracy"] for r in records if r["split"] == split])
                for split in ("train", "validation")}
    counts = Counter(r["split"] for r in records)
    expected = Counter(run["config"]["branch_splits"])
    assert counts == {k: v * run["config"]["snapshots_per_branch"] for k, v in expected.items()}
    assert metrics["complete"]
    summaries = {}
    for same_branch in (True, False):
        group = [p for p in metrics["pairs"] if p["same_branch"] == same_branch]
        if group:
            summaries["within_branch" if same_branch else "across_branch"] = {
                key: describe([p[key] for p in group]) for key in
                ("normalized_weight_rms_distance", "validation_prediction_disagreement")}
    result = {"integrity": "passed", "counts": dict(counts), "accuracy": accuracy,
              "accuracy_gate_passed": all(v["median"] >= .933 for v in accuracy.values()),
              "diversity": summaries, "collection_seconds": metrics["pilot_seconds"],
              "artifact_bytes": sum(p.stat().st_size for p in root.iterdir() if p.is_file())}
    (root / "audit.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    print(json.dumps(audit(parser.parse_args().root), indent=2))
