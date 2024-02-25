import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import keras.backend as K


# Створюєм датасет для навчання
def decode_and_save_rle_vectorized(rle_str, image_id, image_shape=(768, 768)):
    decoded_mask = np.zeros(image_shape, dtype=np.uint8)
    if rle_str != 'nan':
        pairs = np.array(rle_str.split(), dtype=np.int32)
        start = pairs[::2]
        length = pairs[1::2]
        row = start // image_shape[1]
        col = start % image_shape[1]
        for r, c, l in zip(row, col, length):
            decoded_mask[r, c:c + l] = 255
        decoded_mask = decoded_mask.T
        mask_path = f'/home/maxim/masks/{image_id}'
        plt.imsave(mask_path, decoded_mask, cmap='gray')

        image_path = f'/home/maxim/train_v2/{image_id}'
        target_path = f'/home/maxim/images_with_ships/{image_id}'
        try:
            shutil.copyfile(image_path, target_path)
        except FileNotFoundError:
            print(f"File {image_path} not found. Skipping...")
            # Видалення файлу маски, якщо відповідний файл зображення відсутній
            os.remove(mask_path)


chunks = pd.read_csv('/home/maxim/train_ship_segmentations_v2.csv', chunksize=4000)
print('File read')

for i, chunk in enumerate(chunks):
    if i % 2 == 0:
        print(f'Processing {i} chunk')
    grouped_df = chunk.groupby('ImageId')['EncodedPixels'].apply(lambda x: x.str.cat(sep=' ')).reset_index()
    grouped_df['EncodedPixels'] = grouped_df.apply(
        lambda row: decode_and_save_rle_vectorized(row['EncodedPixels'], row['ImageId']), axis=1)






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

