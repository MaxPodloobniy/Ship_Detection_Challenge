import os
import shutil
import pandas as pd


def create_dataset(path_to_data, base_dir, img_path, chunk_size=5000, train_size=50000, valid_size=10000):
    chunks = pd.read_csv(path_to_data, chunksize=chunk_size)
    print('File read')

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.mkdir(base_dir)

    curr_dir = os.path.join(base_dir, 'train/')
    os.mkdir(curr_dir)
    ship_imgs_path = os.path.join(curr_dir, 'ship/')
    not_ship_imgs_path = os.path.join(curr_dir, 'no_ship/')
    os.mkdir(ship_imgs_path)
    os.mkdir(not_ship_imgs_path)

    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            print(f'Processing {i} chunk')

        grouped_df = chunk.groupby('ImageId')['EncodedPixels'].apply(lambda x: x.str.cat(sep=' ')).reset_index()
        grouped_df['EncodedPixels'] = grouped_df.apply(
            lambda row: process_chunk(row['EncodedPixels'], row['ImageId'], img_path, ship_imgs_path,
                                      not_ship_imgs_path), axis=1)

        if chunk_size * i >= train_size+valid_size:
            print(f'Valid dataset created, path {curr_dir}\n process finished')
            break
        elif chunk_size * i >= train_size and curr_dir != os.path.join(base_dir, 'valid/'):
            print(f'Train dataset created, path {curr_dir}')
            curr_dir = os.path.join(base_dir, 'valid/')
            os.mkdir(curr_dir)
            ship_imgs_path = os.path.join(curr_dir, 'ship/')
            not_ship_imgs_path = os.path.join(curr_dir, 'no_ship/')


def process_chunk(rle_str, image_id, image_path, ship_imgs_path, not_ship_imgs_path):
    image = os.path.join(image_path, image_id)
    if rle_str != 'nan' and len(rle_str) != 0:
        target_path = os.path.join(ship_imgs_path, image_id)
    else:
        target_path = os.path.join(not_ship_imgs_path, image_id)
    shutil.copyfile(image, target_path)


# Створюємо датасети для навчання і валідації
create_dataset('path_to_csv', 'Classification_data', 'path_to_images')