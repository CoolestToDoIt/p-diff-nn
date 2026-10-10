from types import SimpleNamespace
import pytest
import torch
from p_diff.diagnose_latents import describe_latent, fit_pca, load_validation_scores, trace_sampling
from p_diff.diffusion import DDPM, Denoiser, DenoiserConfig, DiffusionConfig


def test_pca_is_train_only_and_exposes_off_subspace_samples():
    training = torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
    pca = fit_pca(training)
    assert pca["rank"] == 1
    before = pca["basis"].clone()
    inside = describe_latent(torch.tensor([1., 0.]), training, pca)
    outside = describe_latent(torch.tensor([1., 10.]), training, pca)
    assert inside["pca_residual_rms"] == 0
    assert outside["pca_residual_rms"] > 7
    assert torch.equal(before, pca["basis"])
    assert torch.equal(fit_pca(training)["basis"], before)
    with pytest.raises(ValueError):
        fit_pca(torch.ones(5, 2))


def test_instrumented_sampling_exactly_matches_original_sampler():
    torch.set_num_threads(1)
    denoiser = Denoiser(DenoiserConfig(latent_size=4, width=16, time_width=8)).eval()
    diffusion = DDPM(DiffusionConfig(8, .01, .08))
    generator = SimpleNamespace(denoiser=denoiser, diffusion=diffusion)
    vector, trace = trace_sampling(generator, 123)
    assert torch.equal(vector, diffusion.sample(denoiser, 123)[0])
    assert trace[0][0] == 0
    assert trace[-1][0] == 8
    assert torch.equal(trace[-1][1], vector)


def test_diagnosis_refuses_test_metric_file(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("method,seed,status,validation_accuracy,test_accuracy\ndiffusion,40001,ok,.94,.95\n")
    with pytest.raises(ValueError, match="must not read official test"):
        load_validation_scores(path)
