from picamera2 import Picamera2
from libcamera import Transform
import argparse
from sense_hat import SenseHat
import time
from joblib import load
import numpy as np
from pathlib import Path

sense = SenseHat()
saved = load(Path(__file__).parent / 'baseline_results' / 'random_forest_lower60_avgpool8.joblib')
model = saved['model']
POOL_SIZE = saved['pool_size']

picam2 = Picamera2()

# Combine hflip and vflip for a 180-degree rotation
config = picam2.create_still_configuration(
    transform=Transform(hflip=True, vflip=True)
)
picam2.configure(config)

picam2.start()

CLASSES = ['empty', 'invalid', 'step1', 'step2', 'step3']
i = 0
while True:
    cur_step = CLASSES [i]
    time.sleep(10)

    img = picam2.capture_array('main')

    img = img[int(img.shape[0] * saved['crop_top_fraction']):]
    height, width = img.shape[:2]
    height = height // POOL_SIZE * POOL_SIZE
    width = width // POOL_SIZE * POOL_SIZE
    blocks = img[:height, :width].reshape(
        height // POOL_SIZE, POOL_SIZE, width // POOL_SIZE, POOL_SIZE, 3,
    )
    pooled = blocks.mean(axis=(1, 3), dtype=np.float32)
    pooled = pooled.flatten()

    pred = model.predict(pooled.reshape(1, -1))
    print(pred)

    if pred == cur_step:
        sense.clear((0, 255, 0))
        sense.show_text('Continue')
    else:
        sense.clear((255, 0, 0))
        sense.show_text('Reset')