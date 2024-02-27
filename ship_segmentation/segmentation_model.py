import os
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import keras.backend as K


def get_dataset(img_paths, mask_paths, img_size=(768, 768), batch_size=16):
    """
    Creates a TensorFlow Dataset object from input and output images.

    :param img_paths: Path to directory containing input images.
    :param mask_paths: Path to directory containing output masks
    :param img_size: Size of input images and masks. Defaults to (768, 768).
    :param batch_size: Batch size. Defaults to 16.
    :return: TensorFlow Dataset object containing input-output image pairs.
    """
    def load_data(img_path, mask_path):
        # Load and preprocess input image
        input_img = tf_io.read_file(img_path)
        input_img = tf_io.decode_jpeg(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")

        # Load and preprocess mask
        mask = tf_io.read_file(mask_path)
        mask = tf_io.decode_jpeg(mask, channels=1)
        mask = tf_image.resize(mask, img_size, method="nearest")
        mask = tf_image.convert_image_dtype(mask, "float32")

        return input_img, mask

    # Create dataset from tensor slices and aapply load_data function to each element
    # of dataset, then batch the dataset
    dataset = tf_data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(load_data, num_parallel_calls=tf_data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


img_dir = '/Users/maxim/working_dir/train_v2'
mask_dir = '/Users/maxim/working_dir/masks'

# Generate, sort and slice lists of absolute paths to input images and masks
elements_num = 50000
images_paths = [os.path.join(img_dir, str(filename)) for filename in sorted(os.listdir(img_dir)[:elements_num])]
masks_paths = [os.path.join(mask_dir, str(filename)) for filename in sorted(os.listdir(mask_dir)[:elements_num])]
print('Image paths loaded')

# Slice images and masks path to train and validation lists
val_samples = 10000
train_input_img_paths = images_paths[:-val_samples]
train_target_img_paths = masks_paths[:-val_samples]
val_input_img_paths = images_paths[-val_samples:]
val_target_img_paths = masks_paths[-val_samples:]
print('Paths arrays sliced')

# Create train and validation datasets
train_dataset = get_dataset(train_input_img_paths, train_target_img_paths)
valid_dataset = get_dataset(val_input_img_paths, val_target_img_paths)
print('Datasets created')


# Building modek with U-Net architecture
inputs = keras.Input(shape=(768, 768, 3,))

# Downsampling
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


# Upsampling
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



def dice_coef(y_true, y_pred, smooth=1):
    """
    Compute the Dice coefficient for evaluating the similarity between two sets of binary data.
    The Dice coefficient is computed using the following formula:

        dice = (2 * |y_true ∩ y_pred| + smooth) / (|y_true| + |y_pred| + smooth)

    Where:
    - |y_true ∩ y_pred| is the intersection between y_true and y_pred,
    - |y_true| and |y_pred| are the sums of y_true and y_pred, respectively,
    - smooth is a small constant to avoid division by zero.

    :param y_true: Ground truth binary labels.
    :param y_pred: Predicted binary labels.
    :param smooth: Smoothing factor to avoid division by zero. Default is 1.
    :return: Dice coefficient computed over the batch dimension.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    """
    Combined loss function that combines binary cross-entropy and Dice coefficient for training segmentation models.
    Reference:
    - https://www.kaggle.com/code/kmader/baseline-u-net-model-part-1#Build-a-Model
    - https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/tree/master/loss_functions.py

    :param in_gt: Ground truth binary labels.
    :param in_pred: Predicted binary labels.
    :return: Combined loss value.
    """
    return 1e-3*keras.losses.binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


# Model compilation
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy'])

model.summary()

# Model callbacks
callbacks = [
    # Save the best model.
    ModelCheckpoint("unet_low_filters_model.keras", save_best_only=True, verbose=1),
    # Write logs to TensorBoard
    TensorBoard(log_dir='./logs', histogram_freq=1),
    # Adjust learning rate dynamically.
    ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
                                      patience=3,  verbose=1, mode='max',
                                      epsilon=0.0001, cooldown=2, min_lr=1e-6)
]

# Train the model
model.fit(train_dataset, epochs=30, validation_data=valid_dataset, callbacks=callbacks)
