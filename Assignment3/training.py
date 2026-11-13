import numpy as np
from PIL import Image
import os

empty = Image.open(os.path.join('images', 'empty', 'empty1.jpg'))
step1 = Image.open(os.path.join('images', 'step1', 'step1.jpg'))

new = np.array(step1) - np.array(empty)
im = Image.fromarray(new)
im.save("your_file.jpg")