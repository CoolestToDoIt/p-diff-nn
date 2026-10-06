"""Separate reconstruction inputs from the reserved held-out source branch."""
import argparse
import hashlib
from pathlib import Path
import torch


def reconstruction_subset(source):
    rows = [i for i, r in enumerate(source["records"]) if r["split"] in {"train", "validation"}]
    if not rows:
        raise ValueError("No training/validation vectors")
    return {"vectors": source["vectors"][rows].clone(),
            "normalized_vectors": source["normalized_vectors"][rows].clone(),
            "records": [source["records"][i] for i in rows],
            "normalization_mean": source["normalization_mean"],
            "normalization_std": source["normalization_std"], "manifest": source["manifest"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artifacts/source-full/parameters.pt")
    parser.add_argument("--output", default="artifacts/source-full/reconstruction-inputs.pt")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a fresh output file")
    data = reconstruction_subset(torch.load(args.source, weights_only=True))
    data["source_sha256"] = hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    print(f"Exported {len(data['records'])} training/validation vectors; held-out rows excluded")


if __name__ == "__main__":
    main()
