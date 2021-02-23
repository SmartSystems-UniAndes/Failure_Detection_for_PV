# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 14:00:32 2020

@author: ANDRES FELIPE FLOREZ
"""
from tensorflow import keras
from skimage.io import imread
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import shutil

# Semantic Segmentation path
BASE_PATH = str(os.path.dirname(os.path.abspath('')))

if '\\' in BASE_PATH:
    separator_dir = '\\'
else:
    separator_dir = '/'

path_folder_images = BASE_PATH + separator_dir + 'dataset' + separator_dir + 'images'
path_folder_masks = BASE_PATH + separator_dir + 'dataset' + separator_dir + 'masks'
images_per_photo = 20


def augmentation_data_set(base_path, im_path_folder, masks_path_folder, im_per_photo):
    if '\\' in im_path_folder:
        sep_dir = '\\'
    else:
        sep_dir = '/'

    # Path where the new images will be allocated
    new_path = base_path + sep_dir + 'augmented_train_data'
    new_path_images = 'images'
    new_path_masks = 'masks'

    # Create folders
    try:
        if os.path.exists(new_path + sep_dir + new_path_images):
            shutil.rmtree(new_path + sep_dir + new_path_images)
            shutil.rmtree(new_path + sep_dir + new_path_masks)
        if not (os.path.exists(new_path + sep_dir + new_path_images)):
            os.mkdir(new_path + sep_dir + new_path_images)
            os.mkdir(new_path + sep_dir + new_path_masks)
    except:
        if os.path.exists(new_path + sep_dir + new_path_images):
            shutil.rmtree(new_path + sep_dir + new_path_images)
            shutil.rmtree(new_path + sep_dir + new_path_masks)
        if not (os.path.exists(new_path + sep_dir + new_path_images)):
            os.mkdir(new_path + sep_dir + new_path_images)
            os.mkdir(new_path + sep_dir + new_path_masks)

    # Random Seed
    seed = 54
    np.random.seed(seed)

    # Transform information
    datagen_arg = dict(rotation_range=360,
                       width_shift_range=0.3,
                       height_shift_range=0.1,
                       shear_range=0.2,
                       zoom_range=0.2,
                       horizontal_flip=True,
                       vertical_flip=True,
                       fill_mode='reflect')

    # Preprocessing objects
    image_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_arg)
    mask_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_arg)

    # New dataset
    data_set_images = []
    data_set_masks = []

    # Get original Ids
    images_ids = next(os.walk(im_path_folder))[2]
    masks_ids = next(os.walk(masks_path_folder))[2]

    # Transform images
    for n, id_ in tqdm(enumerate(images_ids), total=len(images_ids)):
        img = imread(im_path_folder + sep_dir + images_ids[n])[:, :, :3]
        img = Image.fromarray(img, 'RGB')
        img2 = imread(masks_path_folder + sep_dir + masks_ids[n])
        data_set_images.append(np.array(img))
        data_set_masks.append(np.array(img2))

    data_set_images = np.array(data_set_images)
    data_set_masks = np.array(data_set_masks)
    data_set_masks = np.expand_dims(data_set_masks, axis=-1)

    # Save images
    i = 0
    for batch in image_datagen.flow(data_set_images, save_to_dir=new_path + sep_dir + new_path_images,
                                    save_prefix='aug',
                                    save_format='png',
                                    batch_size=1, seed=seed
                                    ):
        i += 1
        if i == len(images_ids) * im_per_photo:
            break
    i = 0
    for batch2 in mask_datagen.flow(data_set_masks, save_to_dir=new_path + sep_dir + new_path_masks,
                                    save_prefix='aug',
                                    save_format='png',
                                    batch_size=1, seed=seed
                                    ):
        i += 1
        if i == len(images_ids) * im_per_photo:
            break


# Make data augmentation
augmentation_data_set(BASE_PATH, path_folder_images, path_folder_masks, images_per_photo)
