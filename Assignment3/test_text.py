from sense_hat import SenseHat
import time


sense = SenseHat()

while True:
    sense.show_message('Continue')
    time.sleep(10)