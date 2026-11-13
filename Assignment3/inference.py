from picamera2 import Picamera2
from libcamera import Transform
import argparse
from sense_hat import SenseHat
import time
from joblib import load
import numpy as np

POOL_SIZE = 8

sense = SenseHat()

picam2 = Picamera2()

# Combine hflip and vflip for a 180-degree rotation
config = picam2.create_still_configuration(
    transform=Transform(hflip=True, vflip=True)
)
picam2.configure(config)

picam2.start()

model = load('random_forest.joblib')['model']

steps = ['empty', 'step1', 'step2', 'step3']
i = 0
while True:
    cur_step = steps[i]
    time.sleep(5)

    img = picam2.capture_array('main')
    img = img[int(img.shape[0] * 0.4):]

    height, width = img.shape[:2]
    height = height // POOL_SIZE * POOL_SIZE
    width = width // POOL_SIZE * POOL_SIZE
    blocks = img[:height, :width].reshape(
        height // POOL_SIZE, POOL_SIZE, width // POOL_SIZE, POOL_SIZE, 3,
    )
    pooled = blocks.mean(axis=(1, 3), dtype=np.float32)
    pooled = pooled.flatten()

    pred = model.predict(pooled.reshape(1, -1))
    print(f'Predicted: {pred[0]}, Expected: {cur_step}')

    if pred[0] == cur_step:
        sense.clear((0, 255, 0))
        time.sleep(0.5)
        sense.show_message('Continue')
        i += 1
    elif pred[0] != cur_step or pred[0] == 'step3' == cur_step:
        sense.clear((255, 0, 0))
        time.sleep(0.5)
        sense.show_message('Reset')
        i = 0