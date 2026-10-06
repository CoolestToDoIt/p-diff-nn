"""Manifest-based parameter serialization and training-only normalization."""
from dataclasses import asdict, dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    length: int


class ParameterCodec:
    def __init__(self, model: nn.Module):
        self.specs = []
        offset = 0
        for name, param in model.named_parameters():
            if param.dtype != torch.float32:
                raise ValueError("The initial codec supports float32 parameters only")
            self.specs.append(TensorSpec(name, tuple(param.shape), str(param.dtype), offset, param.numel()))
            offset += param.numel()
        self.size = offset

    def manifest(self):
        return {"version": 1, "size": self.size, "tensors": [asdict(s) for s in self.specs]}

    def _validate(self, model):
        if ParameterCodec(model).manifest() != self.manifest():
            raise ValueError("Architecture does not match parameter manifest")

    def flatten(self, model):
        self._validate(model)
        params = dict(model.named_parameters())
        return torch.cat([params[s.name].detach().cpu().reshape(-1) for s in self.specs])

    def restore(self, vector, model):
        self._validate(model)
        if vector.shape != (self.size,) or vector.dtype != torch.float32:
            raise ValueError("Expected one float32 vector of manifest length")
        if not torch.isfinite(vector).all():
            raise ValueError("Parameter vector contains nonfinite values")
        params = dict(model.named_parameters())
        with torch.no_grad():
            for s in self.specs:
                params[s.name].copy_(vector[s.offset:s.offset+s.length].reshape(s.shape))
        return model


@dataclass
class BlockNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, training_vectors, codec, min_std=1e-6):
        if min_std <= 0 or training_vectors.ndim != 2 or training_vectors.shape[0] == 0:
            raise ValueError("Need nonempty training vectors and positive min_std")
        if training_vectors.shape[1] != codec.size or not torch.isfinite(training_vectors).all():
            raise ValueError("Invalid training vectors")
        mean = torch.empty_like(training_vectors[0])
        std = torch.empty_like(mean)
        for s in codec.specs:
            block = slice(s.offset, s.offset+s.length)
            values = training_vectors[:, block]
            mean[block] = values.mean()
            std[block] = values.std(unbiased=False).clamp_min(min_std)
        return cls(mean, std)

    def normalize(self, vectors):
        return (vectors - self.mean) / self.std

    def inverse(self, vectors):
        return vectors * self.std + self.mean
