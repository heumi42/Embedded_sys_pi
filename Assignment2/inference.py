"""Run on the Pi using a model trained with the same scikit-learn version."""
import argparse
from pathlib import Path
import time

from joblib import load
import numpy as np
import pandas as pd
from sense_hat import SenseHat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).parent / 'model.joblib')
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--batches', type=int, default=0, help='Stop after N predictions; 0 runs continuously.')
    parser.add_argument('--interval', type=float, default=0.05)
    args = parser.parse_args()
    if args.window < 1 or args.batches < 0 or args.interval < 0:
        parser.error('window must be positive; batches and interval must be nonnegative')

    # Load only trusted model files. Joblib, like pickle, requires matching versions.
    model = load(args.model)
    sense = SenseHat()
    completed = 0
    try:
        while args.batches == 0 or completed < args.batches:
            readings = []
            for _ in range(args.window):
                accel = sense.get_accelerometer_raw()
                gyro = sense.get_gyroscope_raw()
                readings.append([accel['x'], accel['y'], accel['z'],
                                 gyro['x'], gyro['y'], gyro['z']])
                time.sleep(args.interval)

            values = np.asarray(readings)
            # Same four per-timestamp features and column order as training.
            features = pd.DataFrame({
                'mean': values.mean(axis=1),
                'std': values.std(axis=1, ddof=0),
                'min': values.min(axis=1),
                'max': values.max(axis=1),
            })
            predictions = model.predict(features)
            labels, counts = np.unique(predictions, return_counts=True)
            # Predictions already contain class names, not numeric class IDs.
            label = labels[counts.argmax()]
            print(f'{label} ({counts.max()}/{len(predictions)} votes)', flush=True)
            completed += 1
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
