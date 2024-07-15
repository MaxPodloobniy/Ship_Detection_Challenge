import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from tensorflow import keras
from PIL import Image


@keras.saving.register_keras_serializable()
def generalized_dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


@keras.saving.register_keras_serializable()
def dice_loss(y_true, y_pred):
    loss = 1 - generalized_dice_coefficient(y_true, y_pred)
    return loss


@keras.saving.register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    loss = keras.losses.binary_crossentropy(y_true, y_pred) + \
           dice_loss(y_true, y_pred)
    return loss / 2.0


ship_detection_model = keras.models.load_model('ship_detection/ship_detection_model.keras')
print('Ship detection model loaded')
segmentation_model_v2 = keras.models.load_model('ship_segmentation/ship_segmentation_model_v2.keras')
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

    segmentation_result = segmentation_model_v2.predict(input_img)
    segmentation_result = tf.squeeze(segmentation_result, axis=0)
    visualize_mask(image_path, segmentation_result)


test_img_dir = '/Users/maxim/airbus-ship-detection/test_v2'
test_img_list = sorted(os.listdir(test_img_dir))[140:160]

for image_id in test_img_list:
    full_path = os.path.join(test_img_dir, image_id)
    predict_and_visualize(full_path)
