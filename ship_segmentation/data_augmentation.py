import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import image as tf_image
from tensorflow import io as tf_io


def vertical_flip(image, mask):
    # Vertical flip
    image_flipped = tf_image.flip_up_down(image)
    mask_flipped = tf_image.flip_up_down(mask)
    return image_flipped, mask_flipped


def horizontal_flip(image, mask):
    # Horizontal flip
    image_flipped = tf_image.flip_left_right(image)
    mask_flipped = tf_image.flip_left_right(mask)
    return image_flipped, mask_flipped


def transpose(image, mask):
    # Transpose
    image_transposed = tf_image.transpose(image)
    mask_transposed = tf_image.transpose(mask)
    return image_transposed, mask_transposed


def adjust_brightness_contrast(image, mask):
    # Adjusts the brightness and contrast of the input image.

    brightness = 0.2
    contrast = 1.0

    image = tf_image.adjust_brightness(image, brightness)
    image = tf_image.adjust_contrast(image, contrast)

    return tf.cast(image, np.uint8), mask


def add_gaussian_noise(image, mask):
    # Adds Gaussian noise to the input image.
    noise_stddev = 0.05 # noise standard deviation

    # Generate Gaussian noise as float32
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_stddev, dtype=tf.float32)

    # Add noise to the image and cast to int32
    noisy_image = tf.cast(image, tf.float32) + noise
    noisy_image = tf.clip_by_value(noisy_image, 0, 255)
    noisy_image = tf.cast(noisy_image, tf.uint8)

    return (noisy_image, mask)


def random_method_chose(input_img, mask):
    methods = [vertical_flip, horizontal_flip, transpose, adjust_brightness_contrast, add_gaussian_noise]
    chosen_method = np.random.choice(methods)
    return chosen_method(input_img, mask)


def augment_and_save_data(base_dir, imgs_dir, masks_dir):
    # Creating base directory for dataset, it must be empty
    if os.path.exists(base_dir):
        user_input = input(
            f"The directory '{base_dir}' already exists. Do you want to remove it and recreate it? (yes/no): ")
        if user_input.lower() == "yes":
            shutil.rmtree(base_dir)
            os.mkdir(base_dir)
            print(f"Directory '{base_dir}' has been removed and recreated.")
        else:
            print("Operation cancelled. Exiting without making any changes.")
    else:
        os.mkdir(base_dir)

    # Creating directories for training data in base dir
    aug_images_dir = os.path.join(base_dir, 'aug_images/')
    os.mkdir(aug_images_dir)
    aug_masks_dir = os.path.join(base_dir, 'aug_masks/')
    os.mkdir(aug_masks_dir)

    # Loading images and masks names
    imgs_list = [os.path.join(imgs_dir, str(filename)) for filename in sorted(os.listdir(imgs_dir))]
    masks_list = [os.path.join(masks_dir, str(filename)) for filename in sorted(os.listdir(masks_dir))]

    if len(imgs_list) != len(masks_list):
        raise ValueError("Lengths of imgs_list and masks_list must be equal")

    counter = 0
    for image_id, mask_id in zip(imgs_list, masks_list):
        input_img = tf_io.read_file(os.path.join(imgs_dir, image_id))
        mask = tf_io.read_file(os.path.join(masks_dir, mask_id))
        input_img = tf_io.decode_jpeg(input_img, channels=3)
        mask = tf_io.decode_jpeg(mask, channels=1)

        input_img = tf.cast(input_img, tf.uint8)
        mask = tf.cast(mask, tf.uint8)

        aug_image, aug_mask = random_method_chose(input_img, mask)

        plt.imsave(os.path.join(aug_images_dir, os.path.basename(image_id)), aug_image)
        plt.imsave(os.path.join(aug_masks_dir, os.path.basename(mask_id)), aug_mask, cmap='gray')

    if counter % 10000 == 0:
        print(f"Processing {counter} image")
    counter += 1


base_directory = '/home/maxim/augmented_data/'
imgs_directory = '/home/maxim/images_with_ships'
masks_directory = '/home/maxim/masks'
augment_and_save_data(base_directory, imgs_directory, masks_directory)

