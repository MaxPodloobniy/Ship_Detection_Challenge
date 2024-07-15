import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from tensorflow import keras
from PIL import Image

segmentation_model_v1 = keras.models.load_model('ship_segmentation/ship_segmentation_model.keras', compile=False)
segmentation_model_v2 = keras.models.load_model('ship_segmentation/ship_segmentation_model_v2.keras', compile=False)
print('Ship segmentation model loaded')


def preprocess_image(img_path):
    input_img = tf_io.read_file(img_path)
    input_img = tf_io.decode_jpeg(input_img, channels=3)
    input_img = tf_image.resize(input_img, (768, 768))
    input_img = tf_image.convert_image_dtype(input_img, "float32")
    input_img = tf.expand_dims(input_img, axis=0)
    return input_img


def visualize_masks(img_path, seg_mask_v1, seg_mask_v2):
    image = Image.open(img_path)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(seg_mask_v1, cmap='gray')
    plt.title('Model v1 Result')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(seg_mask_v2, cmap='gray')
    plt.title('Model v2 Result')
    plt.axis('off')

    plt.show()


def predict_and_visualize(image_path):
    input_img = preprocess_image(image_path)

    seg_result_v1 = segmentation_model_v1.predict(input_img)
    seg_result_v1 = tf.squeeze(seg_result_v1, axis=0)

    seg_result_v2 = segmentation_model_v2.predict(input_img)
    seg_result_v2 = tf.squeeze(seg_result_v2, axis=0)

    visualize_masks(image_path, seg_result_v1, seg_result_v2)


test_img_dir = 'test_imgs'
test_img_list = sorted(os.listdir(test_img_dir))

# test_img_dir = '/Users/maxim/airbus-ship-detection/test_v2'
# test_img_list = sorted(os.listdir(test_img_dir))[160:180]

for image_id in test_img_list:
    full_path = os.path.join(test_img_dir, image_id)
    predict_and_visualize(full_path)
