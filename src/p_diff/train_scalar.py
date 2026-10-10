"""Train and compare the authorized validation-only scalar latent pilot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader
import yaml

from .classifier import Classifier
from .data import mnist_training_data
from .diffusion import DDPM, Denoiser, DenoiserConfig, DiffusionConfig
from .train_diffusion import training_latents
from .scalar_generator import ScalarGenerator, fit_scalar_transform, compress, expand
from .generate_models import make_bundle
from .metrics import classifier_predictions, evaluate_vector, summarize
from .train_autoencoder import load_inputs


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_candidates(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def plot_comparison(output, records, source_median):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    methods = ("scalar_diffusion", "scalar_gaussian", "empirical", "gaussian_128", "latent_mean", "weight_average")
    for index, method in enumerate(methods):
        group = [r for r in records if r["method"] == method and r["status"] == "ok"]
        axes[0].scatter([index]*len(group), [100*r["validation_accuracy"] for r in group], alpha=.7, label=method)
        axes[1].scatter([r["nearest_source_normalized_rms"] for r in group],
                        [100*r["nearest_weight_source_prediction_disagreement"] for r in group], alpha=.7, label=method)
    axes[0].axhline(source_median*100, color="black", linestyle="--", label="Source median")
    axes[0].axhspan((source_median-.02)*100, (source_median+.02)*100, color="green", alpha=.06)
    axes[0].set(xticks=range(len(methods)), xticklabels=["Scalar DDPM", "Scalar normal", "Bootstrap", "128D normal", "Latent mean", "Weights avg"],
                ylabel="Validation accuracy (%)", title="All pilot candidates")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].set(xlabel="Nearest-source normalized weight RMS", ylabel="Prediction disagreement (%)",
                title="Proximity to nearest source by weights")
    axes[1].legend()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scalar_pilot.yaml")
    parser.add_argument("--latents", default="artifacts/autoencoder-pilot/latents.pt")
    parser.add_argument("--autoencoder", default="artifacts/autoencoder-pilot/128/best.pt")
    parser.add_argument("--inputs", default="artifacts/source-full/reconstruction-inputs.pt")
    parser.add_argument("--splits", default="artifacts/source-full/splits.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="artifacts/scalar-pilot")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    expected = {"seed": 42, "max_updates": 2000, "batch_size": 32, "learning_rate": .001,
                "validation_interval": 500, "max_seconds": 1800,
                "tuning_seeds": list(range(60001, 60006)),
                "candidate_seed_start": 70001, "candidate_count": 100}
    if config != expected:
        raise ValueError("Configuration differs from the authorized scalar pilot")
    config["candidate_seeds"] = list(range(70001, 70101))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a fresh pilot directory")
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    latent_data = torch.load(args.latents, weights_only=True)
    autoencoder = torch.load(args.autoencoder, weights_only=True)
    inputs, rows, codec = load_inputs(args.inputs)
    if (not autoencoder.get("gate_passed") or latent_data["autoencoder_sha256"] != sha256(args.autoencoder)
            or autoencoder["input_sha256"] != sha256(args.inputs)
            or latent_data["source_sha256"] != inputs["source_sha256"]
            or autoencoder["source_sha256"] != inputs["source_sha256"]
            or latent_data["records"] != inputs["records"]):
        raise ValueError("Upstream artifact identity mismatch or failed reconstruction gate")
    split_hash = sha256(args.splits)
    if any(r["split_sha256"] != split_hash for r in inputs["records"]):
        raise ValueError("Image split identity mismatch")
    training = training_latents(latent_data)
    if training.shape != (160, 128):
        raise ValueError("Pilot requires 160 training latents of width 128")
    source_vectors = inputs["normalized_vectors"][rows["train"]]
    source_median = statistics.median(inputs["records"][i]["validation_accuracy"] for i in rows["train"])
    output.mkdir(parents=True)
    (output / "models").mkdir()
    started = time.perf_counter()
    deadline = started + config["max_seconds"]
    metadata = {"config": config, "autoencoder_sha256": sha256(args.autoencoder),
                "latents_sha256": sha256(args.latents), "source_sha256": inputs["source_sha256"],
                "split_sha256": split_hash, "source_median_validation_accuracy": source_median,
                "selection_rule": "maximum median tuning validation accuracy; earliest step on ties"}
    metadata["code_sha256"] = {str(path): sha256(path) for path in
        (Path(__file__), Path(__file__).with_name("scalar_generator.py"))}
    (output / "run.json").write_text(json.dumps(metadata, indent=2))
    _, validation = mnist_training_data(args.data_root, torch.load(args.splits, weights_only=True))
    batches = list(DataLoader(validation, batch_size=512))
    images = torch.cat([x for x, _ in batches])
    labels = torch.cat([y for _, y in batches])
    del batches
    transform = fit_scalar_transform(training)
    coefficients = compress(training, transform)
    denoiser = Denoiser(DenoiserConfig(latent_size=1, width=64, blocks=3, time_width=32))
    diffusion = DDPM(DiffusionConfig())
    def bundle_at(step):
        bundle = make_bundle(denoiser, diffusion, autoencoder, latent_data, {**metadata, "step": step})
        bundle.update(scalar_version=1, scalar_transform=transform)
        return bundle
    gate_generator = ScalarGenerator(bundle_at(0))
    gate_rows = []
    for index in rows["validation"]:
        if time.perf_counter() >= deadline:
            write_candidates(output / "projection_gate.csv", gate_rows)
            (output / "results.json").write_text(json.dumps({"complete": False, "status": "cap_during_projection_gate"}, indent=2))
            return
        latent = latent_data["standardized_latents"][index:index+1]
        accuracies = []
        for code in (latent, expand(compress(latent, transform), transform)):
            vector = gate_generator.decode_latent(code)[0]
            pred = classifier_predictions(codec.restore(vector, Classifier()).eval(), images)
            accuracies.append((pred == labels).double().mean().item())
        gate_rows.append({"row": index, "original_accuracy": accuracies[0], "projected_accuracy": accuracies[1]})
    gate = {"original_median": statistics.median(r["original_accuracy"] for r in gate_rows),
            "projected_median": statistics.median(r["projected_accuracy"] for r in gate_rows),
            "first_component_variance": transform["first_component_variance"]}
    gate["median_loss_points"] = 100*(gate["original_median"]-gate["projected_median"])
    gate["passed"] = gate["median_loss_points"] <= 1
    write_candidates(output / "projection_gate.csv", gate_rows)
    (output / "gate.json").write_text(json.dumps(gate, indent=2))
    torch.save(transform, output / "transform.pt")
    print(json.dumps({"projection_gate": gate}), flush=True)
    if not gate["passed"]:
        (output / "results.json").write_text(json.dumps({"complete": False, "status": "projection_gate_failed", "gate": gate}, indent=2))
        return
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=config["learning_rate"])
    noise_generator = torch.Generator().manual_seed(config["seed"])
    history, losses = [], []
    best_accuracy = -1
    completed = 0
    for step in range(1, config["max_updates"] + 1):
        if time.perf_counter() >= deadline:
            break
        denoiser.train()
        clean = coefficients[torch.randint(len(coefficients), (config["batch_size"],), generator=noise_generator)]
        timesteps = torch.randint(diffusion.config.steps, (len(clean),), generator=noise_generator)
        noise = torch.randn(clean.shape, generator=noise_generator)
        noisy = diffusion.add_noise(clean, timesteps, noise)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(denoiser(noisy, timesteps), noise)
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
        completed = step
        losses.append(loss.item())
        if step % config["validation_interval"]:
            continue
        bundle = bundle_at(step)
        generator = ScalarGenerator(bundle)
        accuracies = []
        for seed in config["tuning_seeds"]:
            if time.perf_counter() >= deadline:
                break
            vector = generator.generate_vector(seed)
            model = codec.restore(vector, Classifier()).eval()
            pred = classifier_predictions(model, images)
            accuracies.append((pred == labels).double().mean().item())
        if len(accuracies) != 5:
            break
        median = statistics.median(accuracies)
        history.append({"step": step, "mean_recent_noise_loss": statistics.mean(losses[-100:]),
                        "tuning_seeds": config["tuning_seeds"], "tuning_accuracies": accuracies,
                        "median_tuning_accuracy": median, "elapsed_seconds": time.perf_counter()-started})
        (output / "history.json").write_text(json.dumps(history, indent=2))
        if median > best_accuracy:
            best_accuracy = median
            torch.save(bundle, output / "generator.pt")
        print(json.dumps(history[-1]), flush=True)
    if not (output / "generator.pt").exists():
        torch.save(bundle_at(completed), output / "partial-generator.pt")
        (output / "results.json").write_text(json.dumps({"complete": False, "updates_completed": completed,
                                                       "elapsed_seconds": time.perf_counter()-started}, indent=2))
        return
    generator = ScalarGenerator(torch.load(output / "generator.pt", weights_only=True))
    selection_seconds = time.perf_counter()-started
    source_predictions = []
    for index in rows["train"]:
        if time.perf_counter() >= deadline:
            (output / "results.json").write_text(json.dumps({"complete": False, "updates_completed": completed,
                "status": "cap_during_source_reference_evaluation"}, indent=2))
            return
        model = codec.restore(inputs["vectors"][index], Classifier()).eval()
        source_predictions.append(classifier_predictions(model, images))
    source_predictions = torch.stack(source_predictions)
    candidates, candidate_predictions = [], {}
    requests = [(method, seed) for method in ("scalar_diffusion", "scalar_gaussian", "empirical", "gaussian_128") for seed in config["candidate_seeds"]]
    requests += [("latent_mean", None), ("weight_average", None)]
    for method, seed in requests:
        if time.perf_counter() >= deadline:
            break
        begin = time.perf_counter()
        try:
            rng = torch.Generator().manual_seed(seed or 0)
            bootstrap_row = None
            if method in {"scalar_diffusion", "scalar_gaussian"}:
                coefficient = generator.sample_coefficient(seed, method)
                latent = expand(coefficient, transform)
            elif method == "empirical":
                bootstrap_row = torch.randint(len(coefficients), (1,), generator=rng).item()
                coefficient = coefficients[bootstrap_row:bootstrap_row+1]
                latent = expand(coefficient, transform)
            else:
                latent = torch.randn(1, 128, generator=rng) if method == "gaussian_128" else torch.zeros(1, 128)
                coefficient = compress(latent, transform)
            vector = (inputs["vectors"][rows["train"]].mean(0) if method == "weight_average"
                      else generator.decode_latent(latent)[0])
            generation_seconds = time.perf_counter()-begin
            evaluate_started = time.perf_counter()
            metrics, pred = evaluate_vector(generator, vector, images, labels, source_vectors, source_predictions)
            evaluation_seconds = time.perf_counter()-evaluate_started
            model = codec.restore(vector, Classifier()).eval()
            file = f"models/{method}-{seed if seed is not None else 'single'}.pt"
            torch.save({"state_dict": model.state_dict(), "manifest": codec.manifest(), "seed": seed,
                        "method": method, "generator_sha256": sha256(output / "generator.pt")}, output / file)
            # Verify the complete saved classifier without any optimizer or updates.
            saved = torch.load(output / file, weights_only=True)
            fresh = Classifier().eval()
            fresh.load_state_dict(saved["state_dict"], strict=True)
            assert torch.equal(codec.flatten(fresh), vector)
            record = {"method": method, "seed": seed, "status": "ok", "file": file,
                      "generation_seconds": generation_seconds, "evaluation_seconds": evaluation_seconds, **metrics,
                      "coefficient": coefficient.item() if method != "weight_average" else None,
                      "expanded_latent_norm": latent.norm().item() if method != "weight_average" else None,
                      "nearest_training_latent_rms": (training-latent).square().mean(1).sqrt().min().item() if method != "weight_average" else None,
                      "bootstrap_training_row": bootstrap_row}
            candidate_predictions.setdefault(method, []).append(pred)
        except ValueError as error:
            record = {"method": method, "seed": seed, "status": "failed", "error": str(error)}
        candidates.append(record)
        write_candidates(output / "candidates.csv", candidates)
        print(json.dumps(record), flush=True)
    summary = {method: summarize([r for r in candidates if r["method"] == method], source_median)
               for method in ("scalar_diffusion", "scalar_gaussian", "empirical", "gaussian_128", "latent_mean", "weight_average")
               if any(r["method"] == method for r in candidates)}
    for method, preds in candidate_predictions.items():
        if len(preds) > 1:
            summary[method]["mean_candidate_prediction_disagreement"] = statistics.mean(
                (preds[i] != preds[j]).float().mean().item()
                for i in range(len(preds)) for j in range(i+1, len(preds)))
    for method in summary:
        group = [r for r in candidates if r["method"] == method and r["status"] == "ok"]
        summary[method]["failure_fraction"] = summary[method]["failures"]/summary[method]["count"]
        for field in ("coefficient", "expanded_latent_norm", "nearest_training_latent_rms"):
            values = [r[field] for r in group if r[field] is not None]
            if values:
                summary[method][field+"_range"] = [min(values), max(values)]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"complete": completed == config["max_updates"] and len(candidates) == len(requests),
              "updates_completed": completed, "selected_step": generator.bundle["metadata"]["step"],
              "source_median_validation_accuracy": source_median, "methods": summary,
              "training_and_selection_seconds": selection_seconds, "elapsed_seconds": time.perf_counter()-started,
              "process_peak_rss_bytes": peak if sys.platform == "darwin" else peak*1024,
              "projection_gate": gate, "training_coefficient_min": coefficients.min().item(),
              "training_coefficient_max": coefficients.max().item(),
              "evaluation_scope": "exploratory validation only; no held-out branch or official test"}
    (output / "results.json").write_text(json.dumps(result, indent=2))
    plot_comparison(output, candidates, source_median)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
