import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
'''
# Параметри для завантаження вхідних зображень
input_data_gen = ImageDataGenerator(rescale=1./255)
input_data_flow = input_data_gen.flow_from_directory(
    'шлях/до/папки/вхідних_зображень',
    target_size=(256, 256),
    batch_size=32,
    class_mode=None,  # Ігнорувати мітки, оскільки ми їх не потрібно завантажувати
    shuffle=False)  # Важливо, щоб порядок відповідав порядку масок

# Параметри для завантаження відповідних масок
target_data_gen = ImageDataGenerator(rescale=1./255)
target_data_flow = target_data_gen.flow_from_directory(
    'шлях/до/папки/результатів',
    target_size=(256, 256),
    batch_size=32,
    class_mode=None,  # Ігнорувати мітки
    shuffle=False)  # Важливо, щоб порядок відповідав порядку вхідних зображень


X_train, X_test, y_train, y_test = train_test_split(input_data_flow, target_data_flow, test_size=0.2, random_state=42)
'''
# Побудова моделі U-Net
inputs = keras.Input(shape=(768, 768, 3))

# Зведення (англ. Downsampling)
conv1 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
conv1 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(pool1)
conv2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv2)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool2)
conv3 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv3)
pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool3)
conv4 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool4)
conv5 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

# Перевернення (англ. Upsampling)
up6 = keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
up6 = keras.layers.concatenate([up6, conv4], axis=3)
conv6 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up6)
conv6 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

up7 = keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
up7 = keras.layers.concatenate([up7, conv3], axis=3)
conv7 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up7)
conv7 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

up8 = keras.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv7)
up8 = keras.layers.concatenate([up8, conv2], axis=3)
conv8 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up8)
conv8 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv8)

up9 = keras.layers.Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(conv8)
up9 = keras.layers.concatenate([up9, conv1], axis=3)
conv9 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(up9)
conv9 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(conv9)

outputs = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)

model = keras.Model(inputs, outputs)

# Компіляція моделі
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

'''# Навчання моделі
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Оцінка моделі
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Збереження моделі
model.save('unet_model.h5')
'''