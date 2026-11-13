from picamera2 import Picamera2
from libcamera import Transform
import argparse

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
picam2.capture_file(f"images/{args.filename}.jpg")
picam2.stop()