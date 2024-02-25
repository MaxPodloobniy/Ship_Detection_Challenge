import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil


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
        shutil.copyfile(image_path, target_path)


chunks = pd.read_csv('/home/maxim/train_ship_segmentations_v2.csv', chunksize=4000)
print('File read')

for i, chunk in enumerate(chunks):
    if i % 2 == 0:
        print(f'Processing {i} chunk')
    grouped_df = chunk.groupby('ImageId')['EncodedPixels'].apply(lambda x: x.str.cat(sep=' ')).reset_index()
    grouped_df['EncodedPixels'] = grouped_df.apply(
        lambda row: decode_and_save_rle_vectorized(row['EncodedPixels'], row['ImageId']), axis=1)


