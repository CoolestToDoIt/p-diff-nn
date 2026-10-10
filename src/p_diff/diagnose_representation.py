"""Training/validation-only diagnosis of frozen parameter reconstruction."""
import argparse
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

from .classifier import Classifier
from .data import mnist_training_data
from .diagnose_latents import fit_pca
from .evaluation_protocol import file_hash
from .export_latents import load_autoencoder
from .metrics import classifier_predictions
from .scalar_generator import ScalarGenerator, compress, expand, fit_scalar_transform
from .train_autoencoder import load_inputs, write_csv
from .train_diffusion import training_latents


def variation_metrics(source, reconstructed):
    """Separate mean error from centered variation, using diagnostic group means."""
    if source.shape != reconstructed.shape or source.ndim != 2:
        raise ValueError("Expected matching matrices")
    source, reconstructed = source.double(), reconstructed.double()
    source_mean, reconstructed_mean = source.mean(0), reconstructed.mean(0)
    source_variance = (source-source_mean).square().mean().item()
    reconstructed_variance = (reconstructed-reconstructed_mean).square().mean().item()
    centered_mse = ((source-source_mean)-(reconstructed-reconstructed_mean)).square().mean().item()
    mean_bias_mse = (source_mean-reconstructed_mean).square().mean().item()
    return {"source_variance": source_variance, "reconstructed_variance": reconstructed_variance,
            "variance_ratio": reconstructed_variance/source_variance if source_variance else None,
            "centered_mse": centered_mse, "mean_bias_mse": mean_bias_mse,
            "reconstruction_mse": centered_mse+mean_bias_mse,
            "centered_error_fraction": centered_mse/source_variance if source_variance else None}


def pairwise_disagreement(predictions):
    if len(predictions) < 2:
        return None
    return statistics.mean((predictions[i] != predictions[j]).double().mean().item()
                           for i in range(len(predictions)) for j in range(i+1, len(predictions)))


def pca_summary(pca):
    energy = pca["singular_values"].square()
    cumulative = pca["cumulative_variance"]
    return {"first_component_variance": cumulative[0].item(),
            "rank_99_percent": pca["rank"],
            "rank_999_percent": min(int(torch.searchsorted(cumulative, torch.tensor(.999, dtype=torch.float64)))+1, len(cumulative)),
            "participation_rank": (energy.sum().square()/energy.square().sum()).item(),
            "total_centered_energy": energy.sum().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', default='artifacts/source-full/reconstruction-inputs.pt')
    parser.add_argument('--autoencoder', default='artifacts/autoencoder-pilot/128/best.pt')
    parser.add_argument('--latents', default='artifacts/autoencoder-pilot/latents.pt')
    parser.add_argument('--bundle', default='artifacts/scalar-pilot/generator.pt')
    parser.add_argument('--splits', default='artifacts/source-full/splits.pt')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output', default='artifacts/representation-diagnosis')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh diagnosis directory')
    torch.set_num_threads(1)
    started = time.perf_counter()
    output.mkdir(parents=True)
    def check_cap():
        if time.perf_counter()-started >= 1800:
            (output/'results.json').write_text(json.dumps({'complete': False, 'status': '30_minute_cap'}, indent=2))
            raise TimeoutError('Diagnosis capped; partial evidence retained')
    hashes = {key: file_hash(getattr(args, key)) for key in ('inputs', 'autoencoder', 'latents', 'bundle', 'splits')}
    data, rows, codec = load_inputs(args.inputs)
    checkpoint = torch.load(args.autoencoder, weights_only=True)
    latents = torch.load(args.latents, weights_only=True)
    training = training_latents(latents)
    bundle = torch.load(args.bundle, weights_only=True)
    if (checkpoint['input_sha256'] != hashes['inputs'] or not checkpoint.get('gate_passed')
            or latents['autoencoder_sha256'] != hashes['autoencoder']
            or bundle['metadata']['autoencoder_sha256'] != hashes['autoencoder']
            or bundle['metadata']['latents_sha256'] != hashes['latents']
            or data['records'] != latents['records']
            or any(r['split_sha256'] != hashes['splits'] for r in data['records'])
            or len(rows['train']) != 160 or len(rows['validation']) != 20):
        raise ValueError('Frozen upstream identity mismatch')
    autoencoder = load_autoencoder(checkpoint)
    generator = ScalarGenerator(bundle)
    for name, value in bundle['decoder_state'].items():
        torch.testing.assert_close(value, checkpoint['state_dict'][name], rtol=0, atol=0)
    transform = fit_scalar_transform(training)
    for key in ('mean', 'basis', 'coefficient_mean', 'coefficient_std'):
        torch.testing.assert_close(transform[key], bundle['scalar_transform'][key], rtol=0, atol=0)
    coefficients = compress(training, transform)
    # Define the full probe grid before evaluating any source or probe behavior.
    grid = torch.linspace(coefficients.min().item(), coefficients.max().item(), 21)
    requests = [(f'grid_{i:02d}', value.item()) for i, value in enumerate(grid)] + [('scalar_mean', 0.)]
    run = {'artifact_sha256': hashes, 'training_rows': rows['train'], 'validation_rows': rows['validation'],
           'probes': requests, 'max_seconds': 1800, 'training_updates': 0,
           'evaluation_scope': 'exploratory training/validation only; original official test outcomes already seen',
           'code_sha256': file_hash(__file__)}
    (output/'run.json').write_text(json.dumps(run, indent=2))
    with torch.no_grad():
        encoded = torch.cat([autoencoder.encode(batch) for batch in data['normalized_vectors'].split(16)])
        torch.testing.assert_close(encoded, latents['latents'], rtol=0, atol=0)
        normalized_recon = torch.cat([autoencoder.decode(batch) for batch in encoded.split(16)])
    recon = normalized_recon*data['normalization_std']+data['normalization_mean']
    check_cap()
    weight_pca = fit_pca(data['normalized_vectors'][rows['train']])
    latent_pca = fit_pca(training)
    pcas = {'normalized_weights': pca_summary(weight_pca), 'standardized_codes': pca_summary(latent_pca)}
    torch.save({'normalized_weights': weight_pca, 'standardized_codes': latent_pca}, output/'training_pca.pt')
    pca_rows = [{'representation': name, 'components': i+1, 'cumulative_variance': value.item()}
                for name, pca in [('normalized_weights', weight_pca), ('standardized_codes', latent_pca)]
                for i, value in enumerate(pca['cumulative_variance'])]
    write_csv(output/'pca.csv', pca_rows)
    _, validation = mnist_training_data(args.data_root, torch.load(args.splits, weights_only=True))
    batches = list(DataLoader(validation, batch_size=512))
    images = torch.cat([x for x, _ in batches]); labels = torch.cat([y for _, y in batches])
    del batches
    average = data['vectors'][rows['train']].mean(0)
    average_pred = classifier_predictions(codec.restore(average, Classifier()).eval(), images)
    records, source_preds, recon_preds = [], [], []
    for i, record in enumerate(data['records']):
        check_cap()
        preds = [classifier_predictions(codec.restore(vector, Classifier()).eval(), images)
                 for vector in (data['vectors'][i], recon[i])]
        source_preds.append(preds[0]); recon_preds.append(preds[1])
        original_acc, recon_acc = [(pred == labels).double().mean().item() for pred in preds]
        torch.testing.assert_close(torch.tensor(original_acc), torch.tensor(record['validation_accuracy']), rtol=0, atol=0)
        records.append({'row': i, 'split': record['split'], 'branch_id': record['branch_id'],
                        'source_accuracy': original_acc, 'reconstruction_accuracy': recon_acc,
                        'accuracy_loss_points': 100*(original_acc-recon_acc),
                        'source_reconstruction_disagreement': (preds[0] != preds[1]).double().mean().item(),
                        'source_average_disagreement': (preds[0] != average_pred).double().mean().item(),
                        'reconstruction_average_disagreement': (preds[1] != average_pred).double().mean().item(),
                        'normalized_reconstruction_mse': (data['normalized_vectors'][i]-normalized_recon[i]).square().mean().item()})
        write_csv(output/'checkpoints.csv', records)
    tensor_rows, groups = [], {}
    source_preds, recon_preds = torch.stack(source_preds), torch.stack(recon_preds)
    for split, indices in rows.items():
        check_cap()
        source = data['normalized_vectors'][indices]; decoded = normalized_recon[indices]
        group = [r for r in records if r['split'] == split]
        groups[split] = {**variation_metrics(source, decoded),
                        'median_source_accuracy': statistics.median(r['source_accuracy'] for r in group),
                        'median_reconstruction_accuracy': statistics.median(r['reconstruction_accuracy'] for r in group),
                        'mean_source_reconstruction_disagreement': statistics.mean(r['source_reconstruction_disagreement'] for r in group),
                        'source_pairwise_disagreement': pairwise_disagreement(source_preds[indices]),
                        'reconstruction_pairwise_disagreement': pairwise_disagreement(recon_preds[indices]),
                        'constant_training_mean_mse': (source-data['normalized_vectors'][rows['train']].mean(0)).double().square().mean().item()}
        for spec in codec.specs:
            block = slice(spec.offset, spec.offset+spec.length)
            tensor_rows.append({'split': split, 'tensor': spec.name,
                                **variation_metrics(source[:, block], decoded[:, block])})
        blocks = [r for r in tensor_rows if r['split'] == split]
        for field in ('mean_bias_mse', 'centered_mse', 'reconstruction_mse'):
            groups[split]['balanced_'+field] = statistics.mean(r[field] for r in blocks)
        training_mean = data['normalized_vectors'][rows['train']].mean(0)
        groups[split]['balanced_constant_training_mean_mse'] = statistics.mean(
            (source[:, s.offset:s.offset+s.length]-training_mean[s.offset:s.offset+s.length]).double().square().mean().item()
            for s in codec.specs)
    write_csv(output/'tensor_errors.csv', tensor_rows)
    probes, probe_preds, probe_weights = [], [], []
    for identity, coefficient in requests:
        check_cap()
        latent = expand(torch.tensor([[coefficient]]), transform)
        vector = generator.decode_latent(latent)[0]
        normalized = (vector-data['normalization_mean'])/data['normalization_std']
        pred = classifier_predictions(codec.restore(vector, Classifier()).eval(), images)
        probes.append({'probe': identity, 'coefficient': coefficient,
                       'validation_accuracy': (pred == labels).double().mean().item(),
                       'normalized_rms_to_weight_average': (normalized-data['normalized_vectors'][rows['train']].mean(0)).square().mean().sqrt().item(),
                       'prediction_disagreement_to_weight_average': (pred != average_pred).double().mean().item(),
                       'expanded_latent_norm': latent.norm().item()})
        probe_preds.append(pred); probe_weights.append(normalized)
        write_csv(output/'probes.csv', probes)
    probe_weights = torch.stack(probe_weights)
    sensitivity = {'count': len(probes), 'coefficient_range': [grid[0].item(), grid[-1].item()],
                   'accuracy_range': [min(r['validation_accuracy'] for r in probes), max(r['validation_accuracy'] for r in probes)],
                   'pairwise_prediction_disagreement': pairwise_disagreement(torch.stack(probe_preds)),
                   'normalized_endpoint_rms': (probe_weights[0]-probe_weights[20]).square().mean().sqrt().item(),
                   'normalized_output_variance': (probe_weights-probe_weights.mean(0)).square().mean().item()}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {'complete': True, 'pca': pcas, 'groups': groups, 'sensitivity': sensitivity,
              'weight_average_validation_accuracy': (average_pred == labels).double().mean().item(),
              'elapsed_seconds': time.perf_counter()-started,
              'peak_rss_bytes': peak if sys.platform == 'darwin' else peak*1024, 'training_updates': 0}
    (output/'results.json').write_text(json.dumps(result, indent=2))
    plot(output, pca_rows, tensor_rows, probes)
    # Read-only diagnosis must leave every input byte-identical.
    assert hashes == {key: file_hash(getattr(args, key)) for key in hashes}
    print(json.dumps(result, indent=2))


def plot(output, pca_rows, tensor_rows, probes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for name in ('normalized_weights', 'standardized_codes'):
        group = [r for r in pca_rows if r['representation'] == name]
        axes[0].plot([r['components'] for r in group], [100*r['cumulative_variance'] for r in group], label=name)
    axes[0].set(xscale='log', xlabel='Training PCA components', ylabel='Cumulative variance (%)', title='Source variation versus codes')
    axes[0].legend(fontsize=8)
    tensors = [r for r in tensor_rows if r['split'] == 'train']
    axes[1].bar(range(len(tensors)), [r['mean_bias_mse'] for r in tensors], label='Mean bias')
    axes[1].bar(range(len(tensors)), [r['centered_mse'] for r in tensors], bottom=[r['mean_bias_mse'] for r in tensors], label='Variation error')
    axes[1].set(xticks=range(len(tensors)), xticklabels=[r['tensor'] for r in tensors], ylabel='Normalized MSE', title='Training tensor reconstruction errors')
    axes[1].tick_params(axis='x', rotation=65); axes[1].legend(fontsize=8)
    grid = [r for r in probes if r['probe'].startswith('grid')]
    axes[2].plot([r['coefficient'] for r in grid], [100*r['validation_accuracy'] for r in grid], marker='o')
    axes[2].set(xlabel='Standardized scalar coefficient', ylabel='Validation accuracy (%)', title='Frozen decoder within training range')
    fig.savefig(output/'diagnosis.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
