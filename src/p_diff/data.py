"""Training/validation splits; official test data is intentionally separate."""
import torch
from torch.utils.data import Subset


def split_indices(size=60000, validation_size=10000, seed=42):
    if not 0 < validation_size < size:
        raise ValueError("Validation size must lie between zero and dataset size")
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    return {"seed": seed, "size": size, "train": indices[validation_size:],
            "validation": indices[:validation_size]}


def mnist_training_data(root, splits, download=False):
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    dataset = MNIST(root, train=True, transform=ToTensor(), download=download)
    if len(dataset) != splits["size"]:
        raise ValueError("Split size does not match MNIST training data")
    return tuple(Subset(dataset, splits[key].tolist()) for key in ("train", "validation"))
