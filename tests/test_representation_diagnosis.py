import pytest
import torch
from p_diff.diagnose_representation import variation_metrics, pairwise_disagreement


def test_variation_error_decomposition_distinguishes_bias_and_collapse():
    source = torch.tensor([[1., 3.], [3., 5.]])
    perfect = variation_metrics(source, source)
    assert perfect['reconstruction_mse'] == 0 and perfect['variance_ratio'] == 1
    biased = variation_metrics(source, source+2)
    assert biased['mean_bias_mse'] == 4 and biased['centered_mse'] == 0
    collapsed = variation_metrics(source, source.mean(0).expand_as(source))
    assert collapsed['variance_ratio'] == 0 and collapsed['centered_error_fraction'] == 1
    decoded = torch.tensor([[0., 1.], [1., 1.]])
    metrics = variation_metrics(source, decoded)
    assert metrics['reconstruction_mse'] == (source-decoded).square().double().mean().item()
    with pytest.raises(ValueError):
        variation_metrics(source, decoded[:1])


def test_pairwise_predictions_use_unordered_pairs_once():
    predictions = torch.tensor([[0, 1, 2], [0, 1, 2], [1, 1, 1]])
    assert pairwise_disagreement(predictions) == pytest.approx(4/9)
    assert pairwise_disagreement(predictions[:1]) is None
