"""Generate loadable classifiers from a standalone bundle; no dataset required."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch
from torch import nn

from .autoencoder import AutoencoderConfig
from .classifier import Classifier
from .diffusion import DDPM, Denoiser, DenoiserConfig, DiffusionConfig
from .parameters import ParameterCodec


class FrozenDecoder(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        fields = {key: architecture[key] for key in
                  ("input_size", "latent_size", "channels", "stride", "kernel_size", "padding")}
        fields["channels"] = tuple(fields["channels"])
        self.config = AutoencoderConfig(**fields)
        if self.config.manifest() != architecture or (self.config.stride, self.config.kernel_size, self.config.padding) != (4, 8, 2):
            raise ValueError("Unsupported decoder architecture")
        self.length = architecture["encoded_length"]
        incoming = self.config.channels[-1]
        self.from_latent = nn.Linear(self.config.latent_size, self.length * incoming)
        layers = []
        outgoing_channels = (*reversed(self.config.channels[:-1]), 1)
        for index, outgoing in enumerate(outgoing_channels):
            layers.append(nn.ConvTranspose1d(incoming, outgoing, 8, 4, 2))
            if index < len(outgoing_channels) - 1:
                layers.append(nn.LeakyReLU(.1))
            incoming = outgoing
        self.decoder = nn.Sequential(*layers)

    def forward(self, latents):
        features = self.from_latent(latents).reshape(-1, self.config.channels[-1], self.length)
        return self.decoder(features)[:, 0, :self.config.input_size]


def make_bundle(denoiser, diffusion, autoencoder_checkpoint, latent_data, metadata):
    state = autoencoder_checkpoint["state_dict"]
    return {"version": 1, "denoiser_config": asdict(denoiser.config),
            "denoiser_state": {key: value.detach().cpu().clone() for key, value in denoiser.state_dict().items()},
            "diffusion_config": asdict(diffusion.config), "decoder_architecture": autoencoder_checkpoint["architecture"],
            "decoder_state": {key: value for key, value in state.items()
                              if key.startswith(("from_latent.", "decoder."))},
            "latent_mean": latent_data["latent_mean"], "latent_std": latent_data["latent_std"],
            "normalization_mean": autoencoder_checkpoint["normalization_mean"],
            "normalization_std": autoencoder_checkpoint["normalization_std"],
            "manifest": autoencoder_checkpoint["manifest"], "metadata": metadata}


class Generator:
    def __init__(self, bundle):
        if bundle["version"] != 1:
            raise ValueError("Unsupported generator bundle")
        self.bundle = bundle
        self.denoiser = Denoiser(DenoiserConfig(**bundle["denoiser_config"]))
        self.denoiser.load_state_dict(bundle["denoiser_state"], strict=True)
        self.denoiser.eval().requires_grad_(False)
        self.diffusion = DDPM(DiffusionConfig(**bundle["diffusion_config"]))
        self.decoder = FrozenDecoder(bundle["decoder_architecture"])
        self.decoder.load_state_dict(bundle["decoder_state"], strict=True)
        self.decoder.eval().requires_grad_(False)
        self.codec = ParameterCodec(Classifier())
        if self.codec.manifest() != bundle["manifest"] or self.decoder.config.input_size != self.codec.size:
            raise ValueError("Generator classifier manifest mismatch")
        if self.decoder.config.latent_size != self.denoiser.config.latent_size:
            raise ValueError("Denoiser/decoder latent size mismatch")

    @torch.no_grad()
    def decode(self, standardized):
        latent = standardized * self.bundle["latent_std"] + self.bundle["latent_mean"]
        normalized = self.decoder(latent)
        return normalized * self.bundle["normalization_std"] + self.bundle["normalization_mean"]

    @torch.no_grad()
    def generate_vector(self, seed, method="diffusion"):
        if method == "diffusion":
            standardized = self.diffusion.sample(self.denoiser, seed)
        elif method == "gaussian":
            standardized = torch.randn(1, self.denoiser.config.latent_size,
                                       generator=torch.Generator().manual_seed(seed))
        elif method == "latent_mean":
            standardized = torch.zeros(1, self.denoiser.config.latent_size)
        else:
            raise ValueError("Unknown generation method")
        return self.decode(standardized)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="artifacts/diffusion-pilot/generator.pt")
    parser.add_argument("--seed", type=int, default=30001)
    parser.add_argument("--output", default="artifacts/generated-classifier.pt")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a fresh model output path")
    torch.set_num_threads(1)
    started = time.perf_counter()
    generator = Generator(torch.load(args.bundle, weights_only=True))
    vector = generator.generate_vector(args.seed)
    classifier = generator.codec.restore(vector, Classifier()).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": classifier.state_dict(), "seed": args.seed,
                "manifest": generator.codec.manifest(), "method": "diffusion"}, output)
    print(f"Saved {output} in {time.perf_counter() - started:.3f}s; no data or classifier optimization used")


if __name__ == "__main__":
    main()
