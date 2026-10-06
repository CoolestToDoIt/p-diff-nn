import pytest
import torch
from p_diff.classifier import Classifier
from p_diff.data import split_indices
from p_diff.parameters import BlockNormalizer, ParameterCodec


def test_codec_preserves_tensors_and_predictions():
    torch.manual_seed(7)
    original, restored = Classifier().eval(), Classifier().eval()
    codec = ParameterCodec(original)
    assert codec.size == 25818
    vector = codec.flatten(original)
    codec.restore(vector, restored)
    assert torch.equal(vector, codec.flatten(restored))
    images = torch.rand(5, 1, 28, 28)
    assert torch.equal(original(images), restored(images))
    with pytest.raises(ValueError, match="Architecture"):
        codec.restore(vector, torch.nn.Linear(784, 10))
    with pytest.raises(ValueError):
        codec.restore(vector[:-1], restored)


def test_normalization_round_trip_and_constant_blocks():
    codec = ParameterCodec(Classifier())
    vectors = torch.stack([codec.flatten(Classifier()) for _ in range(3)])
    normalizer = BlockNormalizer.fit(vectors, codec)
    torch.testing.assert_close(normalizer.inverse(normalizer.normalize(vectors)), vectors)
    constant = BlockNormalizer.fit(torch.zeros_like(vectors), codec)
    assert torch.isfinite(constant.normalize(torch.zeros_like(vectors))).all()


def test_split_is_reproducible_disjoint_and_complete():
    first, second = split_indices(), split_indices()
    assert len(first["train"]) == 50000
    assert len(first["validation"]) == 10000
    assert torch.equal(first["train"], second["train"])
    combined = torch.cat([first["train"], first["validation"]])
    assert len(combined.unique()) == 60000
    assert not torch.equal(first["train"], split_indices(seed=43)["train"])
