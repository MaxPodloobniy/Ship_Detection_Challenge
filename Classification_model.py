import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import keras.backend as K


# Побудова моделі U-Net
inputs = keras.Input(shape=(768, 768, 3,))

conv1 = keras.layers.SeparableConv2D(8, 3, activation='relu', padding='same')(inputs)
conv1 = keras.layers.BatchNormalization()(conv1)
conv1 = keras.layers.SeparableConv2D(8, 3, activation='relu', padding='same')(conv1)
conv1 = keras.layers.BatchNormalization()(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = keras.layers.SeparableConv2D(16, 3, activation='relu', padding='same')(pool1)
conv2 = keras.layers.BatchNormalization()(conv2)
conv2 = keras.layers.SeparableConv2D(16, 3, activation='relu', padding='same')(conv2)
conv2 = keras.layers.BatchNormalization()(conv2)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

