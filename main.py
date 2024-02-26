import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from keras.models import load_model
from PIL import Image


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 1e-3*tf.keras.losses.binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


custom_losses = {'dice_p_bce': dice_p_bce,
                 'dice_coef': dice_coef}

ship_detection_model = load_model('/Users/maxim/PycharmProjects/Ship_Detection_Challenge/ship_detection_model.keras')
print('Ship detection model loaded')
unet_low_filters_model = load_model('/Users/maxim/PycharmProjects/Ship_Detection_Challenge/unet_low_filters_model.keras', custom_objects=custom_losses)
print('Ship segmentation model loaded')


def preprocess_image(img_path):
    input_img = tf_io.read_file(img_path)
    input_img = tf_io.decode_jpeg(input_img, channels=3)
    input_img = tf_image.resize(input_img, (768, 768))
    input_img = tf_image.convert_image_dtype(input_img, "float32")
    input_img = tf.expand_dims(input_img, axis=0)
    return input_img


def visualize_mask(img_path, segmentation_mask):
    image = Image.open(img_path)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_mask, cmap='gray')
    plt.title('Results')
    plt.axis('off')

    plt.show()


def predict_and_visualize(image_path):
    input_img = preprocess_image(image_path)
    ship_prediction = ship_detection_model.predict(input_img)

    if ship_prediction > 0.6:
        segmentation_result = unet_low_filters_model.predict(input_img)
        segmentation_result = tf.squeeze(segmentation_result, axis=0)
        visualize_mask(image_path, segmentation_result)
    else:
        print(f'Ships not found on image {image_path}')


test_img_dir = '/Users/maxim/airbus-ship-detection/airbus-ship-detection/test_v2'
test_img_list = sorted(os.listdir(test_img_dir))[:11]

for image_id in test_img_list:
    full_path = os.path.join(test_img_dir, image_id)
    predict_and_visualize(full_path)
