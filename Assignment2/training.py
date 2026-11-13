from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

DATA_DIR = Path(__file__).parent / 'data'
CLASSES = ['idle', 'rotation', 'translation', 'translation_x']
CHANNELS = ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']

train_features, test_features = [], []
y_train, y_test = [], []

for label in CLASSES:
    for iteration in range(1, 13):

        rows = pd.read_csv(
            DATA_DIR / f'{label}{iteration}.csv', usecols=CHANNELS,
        ).head(150)

        if len(rows) < 150:
            raise ValueError(f'Not enough timestamps for {label}{iteration}')

        # Four features across the six sensor readings at each timestamp.
        features = pd.DataFrame({
            'mean': rows.mean(axis=1),
            'std': rows.std(axis=1, ddof=0),
            'min': rows.min(axis=1),
            'max': rows.max(axis=1),
        })
        if iteration <= 8:
            train_features.append(features)
            y_train.extend([label] * len(rows))
        else:
            test_features.append(features)
            y_test.extend([label] * len(rows))

X_train = pd.concat(train_features, ignore_index=True)
X_test = pd.concat(test_features, ignore_index=True)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f'Training timestamps: {len(X_train)}, test timestamps: {len(X_test)}')
print(classification_report(y_test, predictions, zero_division=0))

accuracies = {'overall': accuracy_score(y_test, predictions)}
print('Per-class accuracy (correct / test samples in that class):')
for label in CLASSES:
    total = sum(actual == label for actual in y_test)
    correct = sum(actual == label and predicted == label
                  for actual, predicted in zip(y_test, predictions))
    accuracies[label] = correct / total if total else 0.0
    print(f'{label}: {accuracies[label]:.2%} ({correct}/{total})')

plot_dir = DATA_DIR.parent / 'plots'
plot_dir.mkdir(parents=True, exist_ok=True)
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, labels=CLASSES, cmap='Blues',
    xticks_rotation=30, colorbar=False,
)
plt.title('Random Forest — mean/std/min/max\nConfusion matrix (test timestamps)')
plt.tight_layout()
plt.savefig(plot_dir / 'random_forest_confusion_matrix.png', dpi=180)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
bars = ax.bar(list(accuracies), list(accuracies.values()))
ax.bar_label(bars, labels=[f'{value:.1%}' for value in accuracies.values()])
ax.set(title='Random Forest — mean/std/min/max',
       ylabel='Test accuracy / class recall', ylim=(0, 1.15))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.tick_params(axis='x', rotation=20)
fig.savefig(plot_dir / 'random_forest_accuracy.png', dpi=180)
plt.close(fig)
print(f'Saved plots to {plot_dir}')