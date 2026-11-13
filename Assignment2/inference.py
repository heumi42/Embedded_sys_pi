from sense_hat import SenseHat
import time

sense = SenseHat()
model = None
classes = {"idle": 0, "rotation": 1, "translation": 2, "translation_x": 3}

data = {
	'x_acc': [],
	'y_acc': [],
	'z_acc': [],
	'x_gyro': [],
	'y_gyro': [],
	'z_gyro': []
	}



while True:
    time.sleep(0.005)
    accel = sense.get_accelerometer_raw()
    gyro = sense.get_gyroscope_raw()

    x_acc, y_acc, z_acc = accel['x'], accel['y'], accel['z']
    x_gyro, y_gyro, z_gyro = gyro['x'], gyro['y'], gyro['z']







