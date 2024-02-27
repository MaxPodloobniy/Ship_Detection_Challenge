import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def decode_and_save_rle_vectorized(rle_str, image_id, image_shape=(768, 768)):
    """
    This function decodes the RLE (Run-Length Encoding) encoded string and saves it as an i
    mage mask with name of the corresponding picture.
    :param rle_str: RLE encoded string
    :param image_id: image name
    :param image_shape: shape of the image
    :return: nothing, saves the mask
    """
    decoded_mask = np.zeros(image_shape, dtype=np.uint8)
    # If string isn`t empty, so there is a ship/ships on image
    if rle_str != 'nan' and len(rle_str) != 0:
        pairs = np.array(rle_str.split(), dtype=np.int32)
        start = pairs[::2] # Extract starting positions
        length = pairs[1::2] # Extract lengths of runs
        row = start // image_shape[1] # Calculate row indices using image width
        col = start % image_shape[1] # Calculate column indices using image width
        for r, c, l in zip(row, col, length):
            decoded_mask[r, c:c + l] = 255
        decoded_mask = decoded_mask.T
        # Save decoded mask as image, use only images with ships for model fitting
        mask_path = f'/home/maxim/masks/{image_id}'
        plt.imsave(mask_path, decoded_mask, cmap='gray')


# Read the csv file in chunks
chunks = pd.read_csv('/home/maxim/train_ship_segmentations_v2.csv', chunksize=4000)
print('File read')

for i, chunk in enumerate(chunks):
    if i % 2 == 0:
        print(f'Processing {i} chunk')

    # There is a problem that each ship on image has its own row, so we need to group
    # ships from one image in one row. Then apply decode and save function to each row
    grouped_df = chunk.groupby('ImageId')['EncodedPixels'].apply(lambda x: x.str.cat(sep=' ')).reset_index()
    grouped_df['EncodedPixels'] = grouped_df.apply(
        lambda row: decode_and_save_rle_vectorized(row['EncodedPixels'], row['ImageId']), axis=1)
