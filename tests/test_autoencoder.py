import pytest
import torch

from p_diff.autoencoder import AutoencoderConfig, ParameterAutoencoder, balanced_loss
from p_diff.export_latents import fit_latent_statistics, load_autoencoder
from p_diff.parameters import TensorSpec
from p_diff.prepare_reconstruction import reconstruction_subset
from p_diff.train_autoencoder import load_inputs


def test_model_crop_shapes_gradients_and_saved_architecture():
    torch.set_num_threads(1)
    model = ParameterAutoencoder(AutoencoderConfig(input_size=63, latent_size=4))
    vectors = torch.rand(2, 63)
    assert model.config.padded_size == 256
    assert model.encode(vectors).shape == (2, 4)
    reconstructed = model(vectors)
    assert reconstructed.shape == vectors.shape
    (reconstructed - vectors).square().mean().backward()
    assert model.to_latent.weight.grad is not None
    assert torch.isfinite(model.to_latent.weight.grad).all()
    restored = load_autoencoder({"architecture": model.config.manifest(), "state_dict": model.state_dict()})
    torch.testing.assert_close(model(vectors), restored(vectors), rtol=0, atol=0)
    with pytest.raises(ValueError):
        model(torch.zeros(2, 64))


def test_loss_weights_blocks_equally_and_excludes_padding():
    specs = [TensorSpec("large", (8,), "torch.float32", 0, 8),
             TensorSpec("small", (1,), "torch.float32", 8, 1)]
    target = torch.zeros(2, 16)
    reconstructed = target.clone()
    reconstructed[:, :8] = 1
    reconstructed[:, 8] = 3
    reconstructed[:, 9:] = 1000
    assert balanced_loss(reconstructed, target, specs).item() == 5


def test_latent_statistics_exclude_validation_and_handle_constants():
    latents = torch.tensor([[1., 3.], [3., 3.], [100., -100.]])
    mean, std = fit_latent_statistics(latents, [0, 1])
    latents[2] += 10000
    changed_mean, changed_std = fit_latent_statistics(latents, [0, 1])
    assert torch.equal(mean, changed_mean) and torch.equal(std, changed_std)
    assert torch.isfinite((latents - mean) / std).all()
    torch.testing.assert_close(mean, torch.tensor([2., 3.]))


def test_preparation_excludes_reserved_vectors_and_trainer_rejects_them(tmp_path):
    source = {"vectors": torch.arange(9).reshape(3, 3).float(),
              "normalized_vectors": torch.arange(9).reshape(3, 3).float(),
              "records": [{"split": split} for split in ("train", "validation", "held_out")],
              "normalization_mean": torch.zeros(3), "normalization_std": torch.ones(3), "manifest": {}}
    prepared = reconstruction_subset(source)
    assert prepared["vectors"].shape == (2, 3)
    assert all(r["split"] != "held_out" for r in prepared["records"])
    path = tmp_path / "unfiltered.pt"
    torch.save(source, path)
    with pytest.raises(ValueError, match="held-out"):
        load_inputs(path)
