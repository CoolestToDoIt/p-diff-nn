"""Bounded baseline trainer; synthetic smoke mode needs no downloaded data."""
import argparse
import json
from pathlib import Path
import time

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .classifier import Classifier
from .data import mnist_training_data, split_indices
from .parameters import BlockNormalizer, ParameterCodec


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = count = 0
    for images, labels in loader:
        correct += (model(images).argmax(1) == labels).sum().item()
        count += len(labels)
    return correct / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/foundation.yaml")
    parser.add_argument("--output", default="artifacts/baseline")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    torch.manual_seed(config["seed"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    splits = split_indices(validation_size=config["validation_size"], seed=config["seed"])
    if args.smoke:
        # These random images only check mechanics, never MNIST performance.
        images = torch.rand(96, 1, 28, 28)
        labels = torch.randint(10, (96,))
        train = TensorDataset(images[:64], labels[:64])
        validation = TensorDataset(images[64:], labels[64:])
    else:
        train, validation = mnist_training_data(args.data_root, splits, args.download)
    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(config["seed"]))
    validation_loader = DataLoader(validation, batch_size=config["batch_size"])
    model = Classifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history = []
    start = time.perf_counter()
    for epoch in range(1 if args.smoke else config["epochs"]):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            if time.perf_counter() - start >= args.max_seconds:
                raise TimeoutError("Baseline training budget exhausted")
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        history.append({"epoch": epoch + 1, "loss": total_loss / len(train),
                        "validation_accuracy": accuracy(model, validation_loader)})
    codec = ParameterCodec(model)
    vector = codec.flatten(model)
    restored = codec.restore(vector, Classifier()).eval()
    model.eval()
    probe = next(iter(validation_loader))[0]
    torch.testing.assert_close(model(probe), restored(probe), rtol=0, atol=0)
    normalizer = BlockNormalizer.fit(vector[None], codec)
    torch.testing.assert_close(normalizer.inverse(normalizer.normalize(vector)), vector)
    output.mkdir(parents=True)
    torch.save({"state_dict": model.state_dict(), "seed": config["seed"],
                "config": config, "synthetic": args.smoke}, output / "baseline.pt")
    if not args.smoke:
        torch.save(splits, output / "splits.pt")
    (output / "manifest.json").write_text(json.dumps(codec.manifest(), indent=2))
    metrics = {"synthetic": args.smoke, "history": history,
               "elapsed_seconds": time.perf_counter() - start,
               "parameter_count": codec.size, "round_trip": "passed"}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
