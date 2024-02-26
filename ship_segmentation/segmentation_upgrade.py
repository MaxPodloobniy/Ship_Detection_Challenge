import os
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io

from tensorflow import keras
from keras.models import load_model
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

# Generate and sort lists of absolute paths to input images and masks
images_paths = [os.path.join(img_dir, str(filename)) for filename in sorted(os.listdir(img_dir))]
masks_paths = [os.path.join(mask_dir, str(filename)) for filename in sorted(os.listdir(mask_dir))]
print('Image paths loaded')

# Slice images and masks path to train and validation lists
val_samples = 12000
train_input_img_paths = images_paths[:-val_samples]
train_target_img_paths = masks_paths[:-val_samples]
val_input_img_paths = images_paths[-val_samples:]
val_target_img_paths = masks_paths[-val_samples:]
print('Paths lists sliced')

# Create train and validation datasets
train_dataset = get_dataset(train_input_img_paths, train_target_img_paths)
valid_dataset = get_dataset(val_input_img_paths, val_target_img_paths)
print('Datasets created')


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 1e-3 * keras.losses.binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


custom_losses = {'dice_p_bce': dice_p_bce,
                 'dice_coef': dice_coef}

old_seg_model = load_model('ship_segmentation/unet_low_filters_model.keras', custom_objects=custom_losses)

