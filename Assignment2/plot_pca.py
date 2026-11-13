"""Inspect movement classes with standardized PCA or t-SNE.

Requires numpy, matplotlib, and scikit-learn for t-SNE.
Run: python Assignment2/plot_pca.py [--method tsne]
Each sample uses six sensor channels, excluding the saved CSV index.
Each recording uses channel means and standard deviations (12 features).
The panels are separate unsupervised fits, not classifier scores.
t-SNE emphasizes local neighborhoods; cluster distances/sizes are not reliable.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ('idle', 'rotation', 'translation', 'translation_x')
CHANNELS = ('x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro')
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')


def pca(values):
    """Z-score features, then compute PCA via singular value decomposition."""
    scale = values.std(axis=0)
    standardized = (values - values.mean(axis=0)) / np.where(scale > 0, scale, 1)
    _, singular, components = np.linalg.svd(standardized, full_matrices=False)
    variance = singular ** 2
    if variance.sum() == 0:
        raise ValueError('No variance available for PCA')
    return standardized @ components[:2].T, variance[:2] / variance.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'data')
    parser.add_argument('--method', choices=('pca', 'tsne'), default='pca')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(__file__).parent / 'plots' / f'{args.method}.png'

    samples, sample_labels, recordings, recording_labels, iterations = [], [], [], [], []
    for label in CLASSES:
        for iteration in range(1, 13):
            path = args.data_dir / f'{label}{iteration}.csv'
            with path.open(newline='') as handle:
                values = np.array([
                    [float(row[channel]) for channel in CHANNELS]
                    for row in csv.DictReader(handle)
                ])
            if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
                raise ValueError(f'Invalid or empty sensor data: {path}')
            samples.append(values)
            sample_labels.extend([label] * len(values))
            recordings.append(np.concatenate((values.mean(axis=0), values.std(axis=0))))
            recording_labels.append(label)
            iterations.append(iteration)

    sample_labels = np.array(sample_labels)
    recording_labels = np.array(recording_labels)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    for ax, values, labels, title, is_recording in (
        (axes[0], np.vstack(samples), sample_labels, 'Individual sensor samples (6 channels)', False),
        (axes[1], np.array(recordings), recording_labels, 'Recordings (channel mean + std)', True),
    ):
        if args.method == 'pca':
            scores, variance = pca(values)
            xlabel = f'PC1 ({variance[0]:.1%} explained variance)'
            ylabel = f'PC2 ({variance[1]:.1%} explained variance)'
            print(f'{title}: {len(values)} points; PC1={variance[0]:.1%}, PC2={variance[1]:.1%}, total={variance.sum():.1%}')
        else:
            from sklearn.manifold import TSNE
            from sklearn.preprocessing import StandardScaler

            perplexity = 8 if is_recording else 30
            model = TSNE(n_components=2, perplexity=perplexity, init='pca',
                         learning_rate='auto', random_state=args.seed, max_iter=1500)
            scores = model.fit_transform(StandardScaler().fit_transform(values))
            xlabel, ylabel = 't-SNE 1', 't-SNE 2'
            title += f'\nperplexity={perplexity}, seed={args.seed}'
            print(f'{title}: {len(values)} points; KL divergence={model.kl_divergence_:.4f}')
        for label, color in zip(CLASSES, COLORS):
            selected = labels == label
            ax.scatter(scores[selected, 0], scores[selected, 1], c=color,
                       label=f'{label} (n={selected.sum()})', s=60 if is_recording else 9,
                       alpha=0.85 if is_recording else 0.3, edgecolors='none', rasterized=True)
            if is_recording:
                for index in np.flatnonzero(selected):
                    ax.annotate(str(iterations[index]), scores[index], xytext=(4, 4),
                                textcoords='offset points', fontsize=7, color=color)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
    method_name = 'PCA' if args.method == 'pca' else 't-SNE'
    note = 'Recording labels are iteration numbers; panels use separate fits.'
    if args.method == 'tsne':
        note += '\nLocal neighborhoods only: cluster distances and sizes can be misleading.'
    fig.suptitle(f'Movement classes — standardized {method_name}\n{note}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
