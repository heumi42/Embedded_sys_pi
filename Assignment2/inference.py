from sense_hat import SenseHat
import time
import pickle
import pandas as pd
import numpy as np
from joblib import dump, load


sense = SenseHat()

model = pickle.loads('Assignment2/random_forest.pkl') 


classes = {0: "idle", 1: "rotation", 2: "translation", 3: "translation_x"}

data = {
	'x_acc': [],
	'y_acc': [],
	'z_acc': [],
	'x_gyro': [],
	'y_gyro': [],
	'z_gyro': []
	}

it = 0
preds = []
agg_stats = []
while True:
    accel = sense.get_accelerometer_raw()
    gyro = sense.get_gyroscope_raw()

    x_acc, y_acc, z_acc = accel['x'], accel['y'], accel['z']
    x_gyro, y_gyro, z_gyro = gyro['x'], gyro['y'], gyro['z']

    agg_stats.append([x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro])

    if len(agg_stats) > 10:
        agg_stats = np.asarray(agg_stats)

        features = pd.DataFrame({
              'mean': np.mean(agg_stats, axis=1),
              'std': np.std(agg_stats, axis=1),
               'min': np.min(agg_stats, axis=1),
               'max': np.max(agg_stats, axis=1),
         })

    predictions = model.predict(features)
    pred = np.argmax(np.unique(predictions, return_counts=True)[1])

    print(classes[pred])
