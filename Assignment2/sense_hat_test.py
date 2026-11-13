from sense_hat import SenseHat
import time

sense = SenseHat()

while True:
	accel_only = sense.get_accelerometer_raw()
	pitch, roll, yaw = accel_only['x'], accel_only['y'], accel_only['z']
	print(f"x: {pitch:4f}, y: {roll:.4f}, z: {yaw:.4f}")

	time.sleep(0.1)