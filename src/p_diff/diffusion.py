"""Epsilon-prediction DDPM over standardized parameter latents."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


def timestep_embedding(timesteps, width=64):
    if width < 2 or width % 2:
        raise ValueError("Timestep embedding width must be positive and even")
    frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2, device=timesteps.device) / (width // 2))
    angles = timesteps.float()[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=1)


@dataclass(frozen=True)
class DenoiserConfig:
    latent_size: int = 128
    width: int = 256
    blocks: int = 3
    time_width: int = 64


class Denoiser(nn.Module):
    def __init__(self, config=DenoiserConfig()):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.latent_size, config.width)
        self.time = nn.Sequential(nn.Linear(config.time_width, config.width), nn.SiLU(),
                                  nn.Linear(config.width, config.width))
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
            for _ in range(config.blocks)])
        self.output = nn.Sequential(nn.LayerNorm(config.width), nn.SiLU(),
                                    nn.Linear(config.width, config.latent_size))

    def forward(self, latents, timesteps):
        if latents.ndim != 2 or latents.shape[1] != self.config.latent_size or timesteps.shape != (len(latents),):
            raise ValueError("Invalid denoiser input shapes")
        hidden = self.input(latents) + self.time(timestep_embedding(timesteps, self.config.time_width))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


@dataclass(frozen=True)
class DiffusionConfig:
    steps: int = 200
    beta_start: float = .0005
    beta_end: float = .1


class DDPM(nn.Module):
    def __init__(self, config=DiffusionConfig()):
        super().__init__()
        if config.steps < 2 or not 0 < config.beta_start <= config.beta_end < 1:
            raise ValueError("Require at least two steps and 0 < beta_start <= beta_end < 1")
        self.config = config
        # Compute coefficients in float64, then retain float32 inference buffers.
        beta = torch.linspace(config.beta_start, config.beta_end, config.steps, dtype=torch.float64)
        alpha = 1 - beta
        cumulative = alpha.cumprod(0)
        previous = torch.cat([torch.ones(1, dtype=torch.float64), cumulative[:-1]])
        for name, value in {"beta": beta, "alpha": alpha, "alpha_bar": cumulative,
                            "posterior_variance": beta * (1 - previous) / (1 - cumulative)}.items():
            self.register_buffer(name, value.float())

    def _coefficient(self, values, timesteps):
        if timesteps.dtype != torch.long or not ((timesteps >= 0) & (timesteps < self.config.steps)).all():
            raise ValueError("Timesteps must be valid long indices")
        return values[timesteps][:, None]

    def add_noise(self, clean, timesteps, noise):
        if clean.shape != noise.shape or clean.ndim != 2 or len(clean) != len(timesteps):
            raise ValueError("Invalid forward-noise shapes")
        alpha_bar = self._coefficient(self.alpha_bar, timesteps)
        return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise

    def reverse_step(self, noisy, timesteps, predicted_noise, noise):
        if not noisy.shape == predicted_noise.shape == noise.shape or noisy.ndim != 2 or len(noisy) != len(timesteps):
            raise ValueError("Invalid reverse-noise shapes")
        beta = self._coefficient(self.beta, timesteps)
        alpha = self._coefficient(self.alpha, timesteps)
        cumulative = self._coefficient(self.alpha_bar, timesteps)
        mean = (noisy - beta * predicted_noise / (1 - cumulative).sqrt()) / alpha.sqrt()
        variance = self._coefficient(self.posterior_variance, timesteps)
        return mean + variance.sqrt() * noise  # variance is exactly zero on final step

    @torch.no_grad()
    def sample(self, denoiser, seed, count=1):
        if count < 1:
            raise ValueError("Positive sample count required")
        device = self.beta.device
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(count, denoiser.config.latent_size, generator=generator, device=device)
        denoiser.eval()
        for index in reversed(range(self.config.steps)):
            times = torch.full((count,), index, dtype=torch.long, device=device)
            noise = torch.randn(latents.shape, generator=generator, device=device) if index else torch.zeros_like(latents)
            latents = self.reverse_step(latents, times, denoiser(latents, times), noise)
        if not torch.isfinite(latents).all():
            raise ValueError("Diffusion sampling produced nonfinite latents")
        return latents
