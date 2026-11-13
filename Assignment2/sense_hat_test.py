import argparse
from sense_hat import SenseHat
import time
import pandas as pd

parser = argparse.ArgumentParser(description='Collect Sense HAT movement data.')
parser.add_argument(
	'movement',
	nargs='?',
	default='idle',
	choices=['idle', 'raise', 'rotate', 'translation'],
	help='Movement label used for the CSV filename (default: idle).',
)
args = parser.parse_args()

sense = SenseHat()

idle_timer = 0
eps = 0.05
data = {
	'x_acc': [],
	'y_acc': [],
	'z_acc': [],
	'x_gyro': [],
	'y_gyro': [],
	'z_gyro': []
	}

while True:

	if idle_timer > 40:
		break

	accel = sense.get_accelerometer_raw()
	gyro = sense.get_gyroscope_raw()

	x_acc, y_acc, z_acc = accel['x'], accel['y'], accel['z']
	x_gyro, y_gyro, z_gyro = gyro['x'], gyro['y'], gyro['z']

	if abs(x_gyro) < eps and abs(y_gyro) < eps and abs(z_gyro) < eps:
		print(f'timer increased to {idle_timer}')
		idle_timer += 1
	else:
		idle_timer = 0

	data['x_acc'].append(x_acc)
	data['y_acc'].append(y_acc)
	data['z_acc'].append(z_acc)
	data['x_gyro'].append(x_gyro)
	data['y_gyro'].append(y_gyro)
	data['z_gyro'].append(z_gyro)

	print(f"ACCEL STATS:\nx: {x_acc:4f}, y: {y_acc:.4f}, z: {z_acc:.4f}")
	print(f'GYRO STATS:\nx:{x_gyro:4f}, y: {y_gyro:4f}, z: {z_gyro:4f}')

	time.sleep(0.01)

dataframe = pd.DataFrame(data)
dataframe.to_csv(f'{args.movement}.csv')