import pandas as pd
import glob
import os

train_files = glob.glob(os.path.join('data', '*.csv'))

for f in train_files:
    if 'idle' in f:
        label = 0
    elif 'rotation' in f:
        label = 1
    elif 'translation' in f:
        label = 2
    elif 'translation_z' in f:
        label = 3

    file = pd.read_csv(f)
    file['label'] = label
    file['ts'] = file.index * 0.05
    file.to_csv(f)