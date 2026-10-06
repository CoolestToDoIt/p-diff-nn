"""Deterministic convolutional parameter autoencoder."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AutoencoderConfig:
    input_size: int = 25818
    latent_size: int = 128
    channels: tuple[int, ...] = (8, 16, 32, 32)
    stride: int = 4
    kernel_size: int = 8
    padding: int = 2

    @property
    def padded_size(self):
        multiple = self.stride ** len(self.channels)
        return math.ceil(self.input_size / multiple) * multiple

    def manifest(self):
        return {**asdict(self), "padded_size": self.padded_size,
                "encoded_length": self.padded_size // self.stride ** len(self.channels)}


class ParameterAutoencoder(nn.Module):
    def __init__(self, config=AutoencoderConfig()):
        super().__init__()
        if config.input_size <= 0 or config.latent_size <= 0 or not config.channels:
            raise ValueError("Positive sizes and nonempty channels required")
        if (config.stride, config.kernel_size, config.padding) != (4, 8, 2):
            raise ValueError("This version requires stride 4, kernel 8, padding 2")
        self.config = config
        self.length = config.padded_size // config.stride ** len(config.channels)
        encoder = []
        incoming = 1
        for outgoing in config.channels:
            encoder.extend([nn.Conv1d(incoming, outgoing, 8, 4, 2), nn.LeakyReLU(.1)])
            incoming = outgoing
        self.encoder = nn.Sequential(*encoder)
        self.to_latent = nn.Linear(self.length * incoming, config.latent_size)
        self.from_latent = nn.Linear(config.latent_size, self.length * incoming)
        decoder = []
        outgoing_channels = (*reversed(config.channels[:-1]), 1)
        for i, outgoing in enumerate(outgoing_channels):
            decoder.append(nn.ConvTranspose1d(incoming, outgoing, 8, 4, 2))
            if i < len(outgoing_channels) - 1:
                decoder.append(nn.LeakyReLU(.1))
            incoming = outgoing
        self.decoder = nn.Sequential(*decoder)

    def encode(self, vectors):
        if vectors.ndim != 2 or vectors.shape[1] != self.config.input_size:
            raise ValueError("Input must be a batch of unpadded parameter vectors")
        padded = F.pad(vectors, (0, self.config.padded_size - self.config.input_size))
        return self.to_latent(self.encoder(padded[:, None]).flatten(1))

    def decode(self, latents):
        if latents.ndim != 2 or latents.shape[1] != self.config.latent_size:
            raise ValueError("Latent batch has incorrect shape")
        features = self.from_latent(latents).reshape(-1, self.config.channels[-1], self.length)
        return self.decoder(features)[:, 0, :self.config.input_size]

    def forward(self, vectors):
        return self.decode(self.encode(vectors))


def block_errors(reconstructed, target, specs):
    """Average each real tensor block independently, ignoring any trailing padding."""
    return torch.stack([(reconstructed[:, s.offset:s.offset+s.length] -
                         target[:, s.offset:s.offset+s.length]).square().mean() for s in specs])


def balanced_loss(reconstructed, target, specs):
    return block_errors(reconstructed, target, specs).mean()
