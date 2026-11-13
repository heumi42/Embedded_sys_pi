from picamera2 import Picamera2
from libcamera import Transform
import argparse
from sense_hat import SenseHat
import time


sense = SenseHat()

parser = argparse.ArgumentParser()

parser.add_argument('filename')
args = parser.parse_args()

picam2 = Picamera2()

# Combine hflip and vflip for a 180-degree rotation
config = picam2.create_still_configuration(
    transform=Transform(hflip=True, vflip=True)
)
picam2.configure(config)

picam2.start()

steps = ['empty', 'step1', 'step2', 'step3']
i = 0
while True:
    cur_step = steps[i]
    time.sleep(10)

    img = picam2.capture_array(f"images/{args.filename}.jpg")
    pred = model.predict(img)

    if pred == cur_step:
        sense.show_text('Continue')
    else:
        sense.show_text('Reset')