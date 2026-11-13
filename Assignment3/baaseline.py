"""RGB image baseline: 8x8 average pooling -> flatten -> Random Forest.

Run: python Assignment3/baaseline.py
Requires numpy, Pillow, scikit-learn, matplotlib, and joblib.
"""
from pathlib import Path
import re

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

ROOT = Path(__file__).parent
CLASSES = ['empty', 'invalid', 'step1', 'step2', 'step3']
POOL_SIZE = 8


def extract_features(path):
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert('RGB')
        pixels = np.asarray(image)
        pixels = pixels[int(pixels.shape[0] * 0.4):]
    height, width = pixels.shape[:2]
    height = height // POOL_SIZE * POOL_SIZE
    width = width // POOL_SIZE * POOL_SIZE
    if not height or not width:
        raise ValueError(f'Image too small for pooling: {path}')
    # Non-overlapping pooling blocks, averaged independently per RGB channel.
    blocks = pixels[:height, :width].reshape(
        height // POOL_SIZE, POOL_SIZE, width // POOL_SIZE, POOL_SIZE, 3,
    )
    pooled = blocks.mean(axis=(1, 3), dtype=np.float32)
    return pooled.flatten(), pooled.shape


def main():
    paths, labels, features = [], [], []
    train, test = [], []
    expected_shape = None
    for label in CLASSES:
        files = sorted(
            (ROOT / 'images' / label).glob('*.jpg'),
            key=lambda path: int(re.search(r'(\d+)$', path.stem).group(1)),
        )
        numbers = [int(re.search(r'(\d+)$', path.stem).group(1)) for path in files]
        if numbers != list(range(1, 16)):
            raise ValueError(f'Expected images numbered 1–15 for {label}, got {numbers}')
        for number, path in zip(numbers, files):
            vector, shape = extract_features(path)
            if expected_shape is None:
                expected_shape = shape
            if shape != expected_shape:
                raise ValueError(f'Inconsistent pooled image size: {path}: {shape}')
            (train if number <= 10 else test).append(len(paths))
            paths.append(path)
            labels.append(label)
            features.append(vector)

    X = np.stack(features)
    labels = np.array(labels)
    # Fixed file-level split: images 1–10 train, images 11–15 test per class.
    train, test = np.array(train), np.array(test)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X[train], labels[train])
    predictions = model.predict(X[test])
    print(f'Pooled RGB shape: {expected_shape}; flattened features: {X.shape[1]}')
    print(f'Train images: {len(train)}, test images: {len(test)}')
    print(f'Overall accuracy: {accuracy_score(labels[test], predictions):.2%}')
    print(classification_report(labels[test], predictions, labels=CLASSES, zero_division=0))
    for label in CLASSES:
        mask = labels[test] == label
        correct = int((predictions[mask] == label).sum())
        print(f'{label}: {correct}/{mask.sum()} correct')

    output = ROOT / 'baseline_results'
    output.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        labels[test], predictions, labels=CLASSES, cmap='Blues', colorbar=False,
    )
    plt.title(f'Random Forest — {POOL_SIZE}x{POOL_SIZE} average pooling + flatten')
    plt.tight_layout()
    plt.savefig(output / 'confusion_matrix.png', dpi=180)
    plt.close()
    joblib.dump({
        'model': model, 'pool_size': POOL_SIZE, 'pooled_shape': expected_shape,
        'classes': CLASSES,
    }, output / 'random_forest.joblib', compress=3)
    # Record the exact image split and holdout predictions for inspection.
    import csv
    with (output / 'split.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image', 'class', 'split', 'prediction'])
        for index in train:
            writer.writerow([paths[index].relative_to(ROOT), labels[index], 'train', ''])
        for index, prediction in zip(test, predictions):
            writer.writerow([paths[index].relative_to(ROOT), labels[index], 'test', prediction])
    print(f'Saved model, confusion matrix, and split to {output}')


if __name__ == '__main__':
    main()
