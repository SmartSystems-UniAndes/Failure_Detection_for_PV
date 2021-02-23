# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 12:55:30 2020
@author: ANDRES FELIPE FLOREZ
"""
import os
import numpy as np

from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imread, imsave

# Semantic Segmentation path
BASE_PATH = str(os.path.dirname(os.path.abspath('')))

if '\\' in BASE_PATH:
    separator_dir = '\\'
else:
    separator_dir = '/'

# Random seed
seed = 54
np.random.seed(seed)

# Image dimensions
IMG_WIDTH = 200
IMG_HEIGHT = 200
IMG_CHANNELS = 3

path_folder_images = BASE_PATH + separator_dir + 'dataset' + separator_dir + 'mix' + separator_dir + 'without_adjust' +\
                     separator_dir + 'images'
path_folder_masks = BASE_PATH + separator_dir + 'dataset' + separator_dir + 'mix' + separator_dir + 'without_adjust' +\
                    separator_dir + 'masks'

path_folder_images_adjust = BASE_PATH + separator_dir + 'dataset' + separator_dir + 'mix' + separator_dir +\
                            'with_adjust' + separator_dir + 'images'
path_folder_masks_adjust = BASE_PATH + separator_dir + 'dataset' + separator_dir + 'mix' + separator_dir +\
                           'with_adjust' + separator_dir + 'masks'

Train_images_files = next(os.walk(path_folder_images))[2]
Train_masks_files = next(os.walk(path_folder_masks))[2]

print(Train_images_files[0])
print(Train_images_files[0][0:5])

X_train = np.zeros((len(Train_images_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(Train_images_files), IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)

# Image resizing process
print('Resizing train images')

for n, id_ in tqdm(enumerate(Train_images_files), total=len(Train_images_files)):
    path_image = path_folder_images + '\\'
    path_mask = path_folder_masks + '\\'
    img = imread(path_image + Train_images_files[n])[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    imsave(path_folder_images_adjust + separator_dir + Train_images_files[n], img)
    X_train[n] = img

    mask = imread(path_mask + Train_masks_files[n])

    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    imsave(path_folder_masks_adjust + separator_dir + Train_masks_files[n], mask)
    Y_train[n] = mask
