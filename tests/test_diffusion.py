import copy

import pytest
import torch

from p_diff.autoencoder import AutoencoderConfig, ParameterAutoencoder
from p_diff.diffusion import DDPM, Denoiser, DenoiserConfig, DiffusionConfig, timestep_embedding
from p_diff.generate_models import FrozenDecoder
from p_diff.metrics import summarize
from p_diff.train_diffusion import training_latents


def test_forward_and_reverse_match_closed_form_posterior():
    ddpm = DDPM(DiffusionConfig(8, .01, .08))
    clean = torch.tensor([[.3, -.7], [1., -2.]])
    noise = torch.tensor([[.2, .8], [-.5, .3]])
    times = torch.tensor([0, 5])
    cumulative = ddpm.alpha_bar[times, None]
    noisy = ddpm.add_noise(clean, times, noise)
    torch.testing.assert_close(noisy, cumulative.sqrt()*clean+(1-cumulative).sqrt()*noise)
    previous = torch.cat([torch.ones(1), ddpm.alpha_bar[:-1]])[times, None]
    beta = ddpm.beta[times, None]
    alpha = ddpm.alpha[times, None]
    expected_mean = beta*previous.sqrt()/(1-cumulative)*clean + alpha.sqrt()*(1-previous)/(1-cumulative)*noisy
    reverse_noise = torch.tensor([[100., -100.], [.4, -.8]])
    expected = expected_mean + ddpm.posterior_variance[times, None].sqrt()*reverse_noise
    torch.testing.assert_close(ddpm.reverse_step(noisy, times, noise, reverse_noise), expected, rtol=1e-5, atol=1e-5)
    # Final reverse step must ignore all stochastic reverse noise.
    torch.testing.assert_close(ddpm.reverse_step(noisy[:1], times[:1], noise[:1], reverse_noise[:1]), clean[:1], rtol=1e-5, atol=1e-5)
    assert DDPM().alpha_bar[-1] < 1e-4
    with pytest.raises(ValueError):
        ddpm.add_noise(clean, torch.tensor([-1, 8]), noise)


def test_sampling_is_seeded_finite_and_does_not_consume_global_rng():
    torch.set_num_threads(1)
    denoiser = Denoiser(DenoiserConfig(latent_size=4, width=16, blocks=3, time_width=8))
    ddpm = DDPM(DiffusionConfig(8, .01, .08))
    global_state = torch.random.get_rng_state()
    first = ddpm.sample(denoiser, 123)
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert torch.equal(first, ddpm.sample(denoiser, 123))
    assert not torch.equal(first, ddpm.sample(denoiser, 124))
    assert first.shape == (1, 4) and torch.isfinite(first).all()
    assert not torch.equal(timestep_embedding(torch.tensor([1])), timestep_embedding(torch.tensor([2])))


def test_decoder_bundle_excludes_encoder_and_preserves_decode():
    torch.set_num_threads(1)
    ae = ParameterAutoencoder(AutoencoderConfig(input_size=63, latent_size=4)).eval()
    decoder = FrozenDecoder(ae.config.manifest()).eval()
    decoder.load_state_dict({key: value for key, value in ae.state_dict().items()
                            if key.startswith(("from_latent.", "decoder."))})
    latents = torch.rand(3, 4)
    torch.testing.assert_close(decoder(latents), ae.decode(latents), rtol=0, atol=0)
    assert not any(key.startswith("encoder") for key in decoder.state_dict())


def test_diffusion_training_cannot_use_validation_or_held_out_latents():
    latents = torch.tensor([[1., 2.], [3., 4.], [6., 8.]])
    mean = latents[:2].mean(0)
    std = latents[:2].std(0, unbiased=False)
    data = {"latents": latents, "latent_mean": mean, "latent_std": std,
            "standardized_latents": (latents-mean)/std, "training_rows": [0, 1], "validation_rows": [2],
            "records": [{"split": "train", "branch_id": 0}, {"split": "train", "branch_id": 0},
                        {"split": "validation", "branch_id": 1}]}
    training = training_latents(data)
    changed = copy.deepcopy(data)
    changed["latents"][2] += 100
    changed["standardized_latents"] = (changed["latents"]-mean)/std
    assert torch.equal(training, training_latents(changed))
    changed["records"][2]["split"] = "held_out"
    with pytest.raises(ValueError, match="Held-out"):
        training_latents(changed)


def test_summary_counts_failures_and_uses_same_selection_rule():
    common = {"status": "ok", "generation_seconds": 1., "evaluation_seconds": 2.,
              "nearest_source_normalized_rms": .1, "nearest_weight_source_prediction_disagreement": .01}
    records = [{**common, "seed": 2, "validation_accuracy": .94},
               {**common, "seed": 1, "validation_accuracy": .94},
               {"seed": 3, "status": "failed"}]
    result = summarize(records, .95)
    assert result["validation_selected_seed"] == 1
    assert result["fraction_within_two_points"] == 2/3
    assert result["failures"] == 1 and not result["quality_target_passed"]
