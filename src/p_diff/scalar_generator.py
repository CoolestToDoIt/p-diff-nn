"""Generate a scalar code and expand through train-only PCA and a frozen decoder."""
import argparse
from pathlib import Path
import torch

from .classifier import Classifier
from .diagnose_latents import fit_pca
from .diffusion import DDPM, Denoiser, DenoiserConfig, DiffusionConfig
from .generate_models import FrozenDecoder
from .parameters import ParameterCodec


def fit_scalar_transform(training):
    if training.ndim != 2 or len(training) < 2 or not torch.isfinite(training).all():
        raise ValueError("PCA requires at least two finite training codes")
    pca = fit_pca(training)
    basis = pca["basis"][:1]
    coefficients = (training.double()-pca["mean"]) @ basis.T
    return {"mean": pca["mean"], "basis": basis, "coefficient_mean": coefficients.mean(0),
            "coefficient_std": coefficients.std(0, unbiased=False).clamp_min(1e-6),
            "first_component_variance": pca["cumulative_variance"][0].item()}


def compress(latents, transform):
    coefficient = (latents.double()-transform["mean"]) @ transform["basis"].T
    return ((coefficient-transform["coefficient_mean"])/transform["coefficient_std"]).float()


def expand(coefficients, transform):
    if coefficients.ndim != 2 or coefficients.shape[1] != 1:
        raise ValueError("Scalar coefficients must have shape [batch,1]")
    raw = coefficients.double()*transform["coefficient_std"]+transform["coefficient_mean"]
    return (transform["mean"]+raw @ transform["basis"]).float()


class ScalarGenerator:
    def __init__(self, bundle):
        if bundle["scalar_version"] != 1:
            raise ValueError("Unsupported scalar bundle")
        self.bundle = bundle
        self.transform = bundle["scalar_transform"]
        self.denoiser = Denoiser(DenoiserConfig(**bundle["denoiser_config"]))
        self.denoiser.load_state_dict(bundle["denoiser_state"], strict=True)
        self.denoiser.eval().requires_grad_(False)
        self.diffusion = DDPM(DiffusionConfig(**bundle["diffusion_config"]))
        self.decoder = FrozenDecoder(bundle["decoder_architecture"])
        self.decoder.load_state_dict(bundle["decoder_state"], strict=True)
        self.decoder.eval().requires_grad_(False)
        self.codec = ParameterCodec(Classifier())
        if self.denoiser.config.latent_size != 1 or self.transform["basis"].shape != (1, self.decoder.config.latent_size):
            raise ValueError("Scalar/decoder dimensions disagree")
        if bundle["manifest"] != self.codec.manifest():
            raise ValueError("Classifier manifest mismatch")

    @torch.no_grad()
    def decode_latent(self, standardized):
        if (standardized.ndim != 2 or standardized.shape[1] != self.decoder.config.latent_size
                or not torch.isfinite(standardized).all()):
            raise ValueError("Expected finite expanded latent codes")
        latents = standardized*self.bundle["latent_std"]+self.bundle["latent_mean"]
        normalized = self.decoder(latents)
        vectors = normalized*self.bundle["normalization_std"]+self.bundle["normalization_mean"]
        if not torch.isfinite(vectors).all():
            raise ValueError("Nonfinite generated parameters")
        return vectors

    @torch.no_grad()
    def sample_coefficient(self, seed, method="scalar_diffusion"):
        if method == "scalar_diffusion":
            return self.diffusion.sample(self.denoiser, seed)
        if method == "scalar_gaussian":
            return torch.randn(1, 1, generator=torch.Generator().manual_seed(seed))
        raise ValueError("Unknown scalar sampling method")

    @torch.no_grad()
    def generate_vector(self, seed):
        return self.decode_latent(expand(self.sample_coefficient(seed), self.transform))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="artifacts/scalar-pilot/generator.pt")
    parser.add_argument("--seed", type=int, default=70001)
    parser.add_argument("--output", default="artifacts/scalar-classifier.pt")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Use a fresh output file")
    torch.set_num_threads(1)
    generator = ScalarGenerator(torch.load(args.bundle, weights_only=True))
    model = generator.codec.restore(generator.generate_vector(args.seed), Classifier()).eval()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "manifest": generator.codec.manifest(),
                "seed": args.seed, "method": "scalar_diffusion"}, args.output)
    print(f"Saved {args.output}; no dataset or classifier updates used")


if __name__ == "__main__":
    main()
