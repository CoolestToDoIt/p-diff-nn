"""Evaluate a saved classifier on the fixed validation images only."""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from .classifier import Classifier
from .data import mnist_training_data
from .parameters import ParameterCodec
from .train_baseline import accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--splits", default="artifacts/source-full/splits.pt")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.model, weights_only=True)
    model = Classifier().eval()
    codec = ParameterCodec(model)
    if "manifest" in checkpoint and checkpoint["manifest"] != codec.manifest():
        raise ValueError("Saved classifier manifest mismatch")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    _, validation = mnist_training_data(args.data_root, torch.load(args.splits, weights_only=True))
    started = time.perf_counter()
    value = accuracy(model, DataLoader(validation, batch_size=512))
    print(json.dumps({"model": args.model, "validation_accuracy": value,
                      "validation_count": len(validation), "elapsed_seconds": time.perf_counter()-started}, indent=2))


if __name__ == "__main__":
    main()
