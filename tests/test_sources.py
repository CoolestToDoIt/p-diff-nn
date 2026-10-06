import torch
from p_diff.classifier import Classifier
from p_diff.parameters import ParameterCodec
from p_diff.train_sources import export_dataset


def test_export_fits_only_training_branches_and_restores_validation():
    codec = ParameterCodec(Classifier())
    vectors = [codec.flatten(Classifier()) for _ in range(2)]
    records = [{"branch_id": 0, "split": "train"}, {"branch_id": 1, "split": "validation"}]
    first = export_dataset(vectors, records, codec)
    changed = export_dataset([vectors[0], vectors[1] + 100], records, codec)
    assert torch.equal(first["normalization_mean"], changed["normalization_mean"])
    assert torch.equal(first["normalization_std"], changed["normalization_std"])
    assert first["normalization_fit_rows"].tolist() == [0]
    restored = codec.restore(first["vectors"][1], Classifier())
    assert torch.equal(codec.flatten(restored), vectors[1])


def test_held_out_vectors_cannot_change_normalization():
    codec = ParameterCodec(Classifier())
    vectors = [codec.flatten(Classifier()) for _ in range(3)]
    records = [{"branch_id": i, "split": split} for i, split in
               enumerate(("train", "validation", "held_out"))]
    original = export_dataset(vectors, records, codec)
    changed = export_dataset([vectors[0], vectors[1], vectors[2] + 10], records, codec)
    assert torch.equal(original["normalization_mean"], changed["normalization_mean"])
    assert torch.equal(original["normalization_std"], changed["normalization_std"])
    assert original["normalization_fit_rows"].tolist() == [0]
