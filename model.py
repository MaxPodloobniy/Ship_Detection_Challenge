import os
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from tensorflow import keras


def get_dataset(img_paths, mask_paths, img_size=(768, 768), batch_size=32):
    def load_data(img_path, mask_path):
        input_img = tf_io.read_file(img_path)
        input_img = tf_io.decode_jpeg(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")

        mask = tf_io.read_file(mask_path)
        mask = tf_io.decode_jpeg(mask, channels=1)
        mask = tf_image.resize(mask, img_size, method="nearest")
        mask = tf_image.convert_image_dtype(mask, "uint8")

        return input_img, mask

    dataset = tf_data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(load_data, num_parallel_calls=tf_data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


img_dir = '/Users/maxim/Desktop/train_v2'
mask_dir = '/Users/maxim/Desktop/masks'

# Отримуємо список файлів у папках та сортуємо їх
img_paths = [os.path.join(img_dir, str(filename)) for filename in sorted(os.listdir(img_dir)[:50000])]
mask_paths = [os.path.join(mask_dir, str(filename)) for filename in sorted(os.listdir(mask_dir)[:50000])]
print('Image paths loaded')

val_samples = 10000
train_input_img_paths = img_paths[:-val_samples]
train_target_img_paths = mask_paths[:-val_samples]
val_input_img_paths = img_paths[-val_samples:]
val_target_img_paths = mask_paths[-val_samples:]
print('Paths arrays sliced')

train_dataset = get_dataset(train_input_img_paths, train_target_img_paths)
valid_dataset = get_dataset(val_input_img_paths, val_target_img_paths)
print('Datasets created')


# Побудова моделі U-Net
inputs = keras.Input(shape=(768, 768, 3,))

# Зведення (англ. Downsampling)
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

conv3 = keras.layers.SeparableConv2D(32, 3, activation='relu', padding='same')(pool2)
conv3 = keras.layers.BatchNormalization()(conv3)
conv3 = keras.layers.SeparableConv2D(32, 3, activation='relu', padding='same')(conv3)
conv3 = keras.layers.BatchNormalization()(conv3)
pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = keras.layers.SeparableConv2D(64, 3, activation='relu', padding='same')(pool3)
conv4 = keras.layers.BatchNormalization()(conv4)
conv4 = keras.layers.SeparableConv2D(64, 3, activation='relu', padding='same')(conv4)
conv4 = keras.layers.BatchNormalization()(conv4)
pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = keras.layers.SeparableConv2D(128, 3, activation='relu', padding='same')(pool4)
conv5 = keras.layers.BatchNormalization()(conv5)
conv5 = keras.layers.SeparableConv2D(128, 3, activation='relu', padding='same')(conv5)
conv5 = keras.layers.BatchNormalization()(conv5)
pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)

conv6 = keras.layers.SeparableConv2D(256, 3, activation='relu', padding='same')(pool5)
conv6 = keras.layers.BatchNormalization()(conv6)
conv6 = keras.layers.SeparableConv2D(256, 3, activation='relu', padding='same')(conv6)
conv6 = keras.layers.BatchNormalization()(conv6)


# Перевернення (англ. Upsampling)
up7 = keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv6)
up7 = keras.layers.concatenate([up7, conv5], axis=3)
conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up7)
conv7 = keras.layers.BatchNormalization()(conv7)
conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)
conv7 = keras.layers.BatchNormalization()(conv7)

up8 = keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7)
up8 = keras.layers.concatenate([up8, conv4], axis=3)
conv8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up8)
conv8 = keras.layers.BatchNormalization()(conv8)
conv8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv8)
conv8 = keras.layers.BatchNormalization()(conv8)

up9 = keras.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8)
up9 = keras.layers.concatenate([up9, conv3], axis=3)
conv9 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up9)
conv9 = keras.layers.BatchNormalization()(conv9)
conv9 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)
conv9 = keras.layers.BatchNormalization()(conv9)

up10 = keras.layers.Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(conv9)
up10 = keras.layers.concatenate([up10, conv2], axis=3)
conv10 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(up10)
conv10 = keras.layers.BatchNormalization()(conv10)
conv10 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(conv10)
conv10 = keras.layers.BatchNormalization()(conv10)

up11 = keras.layers.Conv2DTranspose(8, 2, strides=(2, 2), padding='same')(conv10)
up11 = keras.layers.concatenate([up11, conv1], axis=3)
conv11 = keras.layers.Conv2D(8, 3, activation='relu', padding='same')(up11)
conv11 = keras.layers.BatchNormalization()(conv11)
conv11 = keras.layers.Conv2D(8, 3, activation='relu', padding='same')(conv11)
conv11 = keras.layers.BatchNormalization()(conv11)

outputs = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv11)

model = keras.Model(inputs, outputs)

# Компіляція моделі
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# Learning Rate Scheduler
def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.9
    else:
        return lr


callbacks = [
    keras.callbacks.ModelCheckpoint("unet_low_filters_model.keras", save_best_only=True),
    keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
    keras.callbacks.LearningRateScheduler(lr_scheduler)
]

# Навчання моделі
#model.fit(train_dataset, epochs=30, validation_data=valid_dataset, callbacks=callbacks)