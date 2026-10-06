"""Export frozen autoencoder latents, standardizing from training rows only."""
import argparse
import hashlib
from pathlib import Path
import torch

from .autoencoder import AutoencoderConfig, ParameterAutoencoder
from .train_autoencoder import load_inputs


def fit_latent_statistics(latents, training_rows):
    training = latents[training_rows]
    if training.ndim != 2 or len(training) == 0 or not torch.isfinite(training).all():
        raise ValueError("Finite nonempty training latents required")
    return training.mean(0), training.std(0, unbiased=False).clamp_min(1e-6)


def load_autoencoder(checkpoint):
    architecture = checkpoint["architecture"]
    fields = {key: architecture[key] for key in
              ("input_size", "latent_size", "channels", "stride", "kernel_size", "padding")}
    fields["channels"] = tuple(fields["channels"])
    config = AutoencoderConfig(**fields)
    if config.manifest() != architecture:
        raise ValueError("Saved shape configuration is inconsistent")
    model = ParameterAutoencoder(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", default="artifacts/source-full/reconstruction-inputs.pt")
    parser.add_argument("--output", default="artifacts/autoencoder-pilot/latents.pt")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a fresh output file")
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    if not checkpoint.get("gate_passed"):
        raise ValueError("Reconstruction accuracy gate must pass before latent export")
    if checkpoint["input_sha256"] != hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest():
        raise ValueError("Reconstruction input identity mismatch")
    data, rows, codec = load_inputs(args.inputs)
    if checkpoint["manifest"] != codec.manifest():
        raise ValueError("Checkpoint manifest mismatch")
    for key in ("normalization_mean", "normalization_std"):
        torch.testing.assert_close(checkpoint[key], data[key], rtol=0, atol=0)
    model = load_autoencoder(checkpoint)
    with torch.no_grad():
        latents = torch.cat([model.encode(batch) for batch in data["normalized_vectors"].split(16)])
        repeat = torch.cat([model.encode(batch) for batch in data["normalized_vectors"].split(16)])
    torch.testing.assert_close(latents, repeat, rtol=0, atol=0)
    if not torch.isfinite(latents).all():
        raise ValueError("Nonfinite exported latents")
    mean, std = fit_latent_statistics(latents, rows["train"])
    standardized = (latents - mean) / std
    torch.testing.assert_close(standardized * std + mean, latents)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": latents, "standardized_latents": standardized,
                "latent_mean": mean, "latent_std": std, "records": data["records"],
                "training_rows": rows["train"], "validation_rows": rows["validation"],
                "autoencoder_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                "source_sha256": checkpoint["source_sha256"]}, output)
    print(f"Exported deterministic latents {list(latents.shape)}; statistics fit on {len(rows['train'])} training rows")


if __name__ == "__main__":
    main()
