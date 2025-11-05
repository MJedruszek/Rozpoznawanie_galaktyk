import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

image_size = (180, 180)
batch_size = 128
# path = "C:\\Users\\gosia\\Dokumenty\\Nauka\\VII sem\\Inżynierka\\canny"
# path = "\\canny"
path = "C:\\Users\\gosia\\Dokumenty\\Nauka\\tst"

if os.path.exists(path):
    # Proceed with data loading
    print("Path found")
else:
    print("File not found.")

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    directory=path,
    validation_split=0.3,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size
)