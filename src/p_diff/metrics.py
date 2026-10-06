"""Validation-only quality and proximity metrics for the exploratory pilot."""
import statistics
import torch
from .classifier import Classifier


@torch.no_grad()
def classifier_predictions(model, images, batch_size=512):
    model.eval()
    return torch.cat([model(batch).argmax(1) for batch in images.split(batch_size)])


@torch.no_grad()
def evaluate_vector(generator, vector, images, labels, source_vectors, source_predictions):
    model = generator.codec.restore(vector, Classifier()).eval()
    predictions = classifier_predictions(model, images)
    normalized = (vector - generator.bundle["normalization_mean"]) / generator.bundle["normalization_std"]
    distances = (source_vectors - normalized[None]).square().mean(1).sqrt()
    nearest = distances.argmin().item()
    disagreement = (source_predictions != predictions[None]).float().mean(1)
    return {"validation_accuracy": (predictions == labels).double().mean().item(),
            "nearest_source_row": nearest, "nearest_source_normalized_rms": distances[nearest].item(),
            "nearest_weight_source_prediction_disagreement": disagreement[nearest].item(),
            "minimum_source_prediction_disagreement": disagreement.min().item(),
            "mean_source_prediction_disagreement": disagreement.mean().item()}, predictions


def summarize(records, source_median):
    successful = [r for r in records if r["status"] == "ok"]
    values = [r["validation_accuracy"] for r in successful]
    if not values:
        return {"count": len(records), "failures": len(records), "quality_target_passed": False}
    selected = max(successful, key=lambda r: (r["validation_accuracy"], -(r["seed"] or 0)))
    within = sum(abs(value - source_median) <= .02 for value in values) / len(records)
    return {"count": len(records), "failures": len(records) - len(successful),
            "mean_accuracy": statistics.mean(values), "median_accuracy": statistics.median(values),
            "std_accuracy": statistics.pstdev(values), "worst_accuracy": min(values),
            "best_accuracy": max(values), "fraction_within_two_points": within,
            "quality_target_passed": abs(statistics.median(values) - source_median) <= .02 and within >= .8,
            "validation_selected_seed": selected["seed"],
            "validation_selected_accuracy": selected["validation_accuracy"],
            "mean_generation_seconds": statistics.mean(r["generation_seconds"] for r in successful),
            "mean_evaluation_seconds": statistics.mean(r["evaluation_seconds"] for r in successful),
            "mean_nearest_source_normalized_rms": statistics.mean(r["nearest_source_normalized_rms"] for r in successful),
            "mean_nearest_weight_source_disagreement": statistics.mean(r["nearest_weight_source_prediction_disagreement"] for r in successful)}
