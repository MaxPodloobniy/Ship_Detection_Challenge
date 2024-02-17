import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Зчитуємо дані з файлу CSV
df = pd.read_csv('/Users/maxim/airbus-ship-detection/airbus-ship-detection/train_ship_segmentations_v2.csv')
print('File read')


# Function for decoding Run Length Encoding
def decode_rle(rle_str, img_shape):
    # Ініціалізуємо пустий масив для розкодованої маски
    decoded_mask = np.zeros(img_shape, dtype=np.uint8)

    if rle_str == 'nan':
        return decoded_mask
    else:
        # Розбиваємо рядок RLE
        pairs = rle_str.split()

        # Ітеруємося по парам (start, length) та встановлюємо відповідні значення
        for i in range(0, len(pairs), 2):
            start = int(pairs[i])
            length = int(pairs[i + 1])

            # Розраховуємо розміщення пікселів у двовимірному масиві
            row = start // img_shape[1]  # Рядок у двовимірному масиві
            col = start % img_shape[1]  # Стовпець у двовимірному масиві

            # Встановлюємо значення пікселів у відповідних позиціях
            decoded_mask[row, col:col + length] = 1  # Встановлюємо значення пікселів на 255 (білий)

        return decoded_mask.T


def decode_rle_vectorized(rle_str, image_shape=(768, 768)):
    decoded_image = np.zeros(image_shape, dtype=np.uint8)
    if rle_str != 'nan':
        pairs = np.array(rle_str.split(), dtype=np.int32) - 1
        start = pairs[::2]
        length = pairs[1::2]
        row = start // image_shape[1]
        col = start % image_shape[1]
        for r, c, l in zip(row, col, length):
            decoded_image[r, c:c + l] = 255
    return decoded_image.T


chunks = pd.read_csv('/Users/maxim/airbus-ship-detection/airbus-ship-detection/train_ship_segmentations_v2.csv',
                     chunksize=4000)
for i, chunk in enumerate(chunks):
    print('Entered cycle')
    # Обробка кожної частини
    grouped_df = chunk.groupby('ImageId')['EncodedPixels'].apply(lambda x: x.str.cat(sep=' ')).reset_index()
    grouped_df['EncodedPixels'] = grouped_df['EncodedPixels'].apply(lambda x: decode_rle_vectorized(x, (768, 768)))
    grouped_df.to_csv(f'mask_chunks/processed_chunk_{i}.csv', index=False)
    print('Exit cycle')



