import os
import shutil
import pandas as pd


def create_dataset(path_to_data, base_dir, img_path, chunk_size=5000, train_size=50000, valid_size=10000):
    """
    Create a dataset by processing data from a CSV file

    :param path_to_data: Path to the CSV file containing mask annotations.
    :param base_dir: Base directory where the dataset will be created.
    :param img_path: Path to the directory containing the input images.
    :param chunk_size: Size of chunks to read from the CSV file. Default is 5000.
    :param train_size: Size of the training dataset to be created. Default is 50000.
    :param valid_size: Size of the validation dataset to be created. Default is 10000.
    :return: None
    """
    chunks = pd.read_csv(path_to_data, chunksize=chunk_size)
    print('File read')

    # Creating base directory for dataset it must be empty
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.mkdir(base_dir)

    # Creating directories for training data in base dir
    curr_dir = os.path.join(base_dir, 'train/')
    os.mkdir(curr_dir)
    ship_imgs_path = os.path.join(curr_dir, 'ship/')
    not_ship_imgs_path = os.path.join(curr_dir, 'no_ship/')
    os.mkdir(ship_imgs_path)
    os.mkdir(not_ship_imgs_path)

    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            print(f'Processing {i} chunk')

        # There is a problem that each ship on image has its own row, so we need to group
        # ships from one image in one row. Then apply decode and save function to each row
        grouped_df = chunk.groupby('ImageId')['EncodedPixels'].apply(lambda x: x.str.cat(sep=' ')).reset_index()
        grouped_df['EncodedPixels'] = grouped_df.apply(
            lambda row: process_image(row['EncodedPixels'], row['ImageId'], img_path,
                                      ship_imgs_path, not_ship_imgs_path), axis=1)


        if chunk_size * i > train_size+valid_size:
            # If the accumulated data size exceeds the sum of required training and validation sizes,
            # the dataset creation process had been completed.

            print(f'Valid dataset created, path {curr_dir}\n Process finished')
            break
        elif chunk_size * i > train_size and curr_dir != os.path.join(base_dir, 'valid/'):
            # If the accumulated data size exceeds the training dataset size and this is the first occurrence,
            # create a directory for the validation dataset and switch to it as the current directory.

            print(f'Train dataset created, path {curr_dir}')
            curr_dir = os.path.join(base_dir, 'valid/')
            os.mkdir(curr_dir)
            ship_imgs_path = os.path.join(curr_dir, 'ship/')
            not_ship_imgs_path = os.path.join(curr_dir, 'no_ship/')


def process_image(rle_str, image_id, image_path, ship_imgs_path, not_ship_imgs_path):
    """
    Process RLE for an image and copy image to appropriate(ship/, no_ship/) directories.

    :param rle_str: Run-length encoded string representing annotations for an image.
    :param image_id: Input image name
    :param image_path: Path to the directory containing the input images.
    :param ship_imgs_path: Path to the directory where images containing ships
    :param not_ship_imgs_path: Path to the directory where images not containing ships
    :return: None
    """
    image = os.path.join(image_path, image_id)
    if rle_str != 'nan' and len(rle_str) != 0:
        target_path = os.path.join(ship_imgs_path, image_id)
    else:
        target_path = os.path.join(not_ship_imgs_path, image_id)
    shutil.copyfile(image, target_path)


# Create validation and train datasets
create_dataset('/home/maxim/working_dir/train_ship_segmentations_v2.csv',
               '/home/maxim/working_dir/classification_data',
               '/hame/maxim/working_dir/train_v2')
