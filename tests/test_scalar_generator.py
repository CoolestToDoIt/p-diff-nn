import torch
import pytest

from p_diff.scalar_generator import fit_scalar_transform, compress, expand, ScalarGenerator
from p_diff.classifier import Classifier
from p_diff.parameters import ParameterCodec
from p_diff.autoencoder import ParameterAutoencoder, AutoencoderConfig
from p_diff.diffusion import Denoiser, DenoiserConfig, DDPM, DiffusionConfig
from p_diff.generate_models import make_bundle


def test_rank_one_roundtrip_standardization_and_unbounded_expansion():
    training = torch.arange(-5., 6.)[:, None]*torch.tensor([[2., 3., -1.]])+7
    transform = fit_scalar_transform(training)
    coefficients = compress(training, transform)
    torch.testing.assert_close(coefficients.mean(0), torch.zeros(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(coefficients.std(0, unbiased=False), torch.ones(1))
    torch.testing.assert_close(expand(coefficients, transform), training)
    extrapolated = coefficients[:1]*100
    torch.testing.assert_close(compress(expand(extrapolated, transform), transform), extrapolated)
    assert expand(extrapolated, transform).norm() > training.norm(dim=1).max()
    with pytest.raises(ValueError):
        expand(torch.ones(2, 3), transform)


def test_projection_removes_orthogonal_variation_without_refitting():
    training = torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
    transform = fit_scalar_transform(training)
    validation = torch.tensor([[8., 999.]])
    torch.testing.assert_close(expand(compress(validation, transform), transform), torch.tensor([[8., 0.]]))
    again = fit_scalar_transform(training)
    for key in ('mean', 'basis', 'coefficient_mean', 'coefficient_std'):
        assert torch.equal(transform[key], again[key])


def test_scalar_bundle_is_standalone_frozen_and_seeded():
    torch.set_num_threads(1)
    codec = ParameterCodec(Classifier())
    ae = ParameterAutoencoder(AutoencoderConfig(input_size=codec.size, latent_size=4)).eval()
    denoiser = Denoiser(DenoiserConfig(latent_size=1, width=16, blocks=3, time_width=8))
    training = torch.arange(-2., 3.)[:, None]*torch.tensor([[1., 2., 3., 4.]])
    transform = fit_scalar_transform(training)
    latent_data = {'latent_mean': torch.zeros(4), 'latent_std': torch.ones(4)}
    checkpoint = {'state_dict': ae.state_dict(), 'architecture': ae.config.manifest(),
                  'normalization_mean': torch.zeros(codec.size), 'normalization_std': torch.ones(codec.size),
                  'manifest': codec.manifest()}
    bundle = make_bundle(denoiser, DDPM(DiffusionConfig(8, .01, .08)), checkpoint, latent_data, {})
    bundle.update(scalar_version=1, scalar_transform=transform)
    generator = ScalarGenerator(bundle)
    assert not any(k.startswith('encoder') for k in bundle['decoder_state'])
    assert not any(p.requires_grad for p in generator.decoder.parameters())
    latent = expand(torch.tensor([[.5]]), transform)
    torch.testing.assert_close(generator.decode_latent(latent), ae.decode(latent), rtol=0, atol=0)
    state = torch.random.get_rng_state()
    assert torch.equal(generator.generate_vector(123), generator.generate_vector(123))
    assert torch.equal(state, torch.random.get_rng_state())
    assert not torch.equal(generator.generate_vector(123), generator.generate_vector(124))
