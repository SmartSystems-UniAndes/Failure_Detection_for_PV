# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:29:54 2020
@author: ANDRES FELIPE FLOREZ
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.io import imread, imshow
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

# Random seed
seed = 54
np.random.seed(seed)

# Image dimensions
IMG_WIDTH = 200
IMG_HEIGHT = 200
IMG_CHANNELS = 3

# Dataset path
BASE_PATH = str(os.path.dirname(os.path.abspath('')))

if '\\' in BASE_PATH:
    separator_dir = '\\'
else:
    separator_dir = '/'

TRAIN_PATH_IMAGES = BASE_PATH + separator_dir + 'dataset' + separator_dir + 'mix' + separator_dir + 'with_adjust' +\
                    separator_dir + 'images'
TRAIN_PATH_MASKS = BASE_PATH + separator_dir + 'dataset' + separator_dir + 'mix' + separator_dir + 'with_adjust' +\
                   separator_dir + 'masks'

# Get images ids
Train_images_files = next(os.walk(TRAIN_PATH_IMAGES))[2]
Train_masks_files = next(os.walk(TRAIN_PATH_MASKS))[2]

X_train = np.zeros((len(Train_images_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(Train_images_files), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
Y_train_class = np.zeros((len(Train_images_files), 1), dtype=np.uint8)

# Image resizing images
print("Resizing train images...")

for n, id_ in tqdm(enumerate(Train_images_files), total=len(Train_images_files)):
    path_image = TRAIN_PATH_IMAGES + '\\'
    path_mask = TRAIN_PATH_MASKS + '\\'
    img = imread(path_image + Train_images_files[n])[:, :, :IMG_CHANNELS]
    X_train[n] = img

    mask = imread(path_mask + Train_masks_files[n])
    mask = np.expand_dims(mask, axis=-1)
    Y_train[n] = mask
    X_train[n] = X_train[n] * Y_train[n]

    name = id_[:5]
    if name == 'N_C_F':
        Y_train_class[n] = 1
    if name == 'N_C_S':
        Y_train_class[n] = 2

    if name == 'N_C_T':
        Y_train_class[n] = 3

    if name == 'N_S_S':
        Y_train_class[n] = 0

Y_train_class = to_categorical(Y_train_class, num_classes=4)

# Load Model
model_name = os.listdir('models')
model = tf.keras.models.load_model('models' + separator_dir + model_name[-1])

# Make a prediction
preds_train = model.predict(X_train, verbose=1)

# Perform a sanity check on some random validation samples
ix = 5
imshow(X_train[ix])
plt.xticks([])
plt.yticks([])
plt.show()
print('Prediction: ', preds_train[ix])
print('True: ', Y_train_class[ix])

# Deconvolution process
weights = model.get_weights()

# Input layer
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

# Contraction path
l0 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(inputs)
l1 = tf.keras.layers.BatchNormalization(axis=1)(l0)
l2 = tf.keras.layers.MaxPooling2D((2, 2))(l1)
l3 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(l2)
l4 = tf.keras.layers.BatchNormalization(axis=1)(l3)
l5 = tf.keras.layers.MaxPooling2D((2, 2))(l4)
l6 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(l5)
l7 = tf.keras.layers.BatchNormalization(axis=1)(l6)
l8 = tf.keras.layers.MaxPooling2D((2, 2))(l7)
l9 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(l8)
l10 = tf.keras.layers.BatchNormalization(axis=1)(l9)
l11 = tf.keras.layers.MaxPooling2D((2, 2))(l10)
l12 = tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(l11)
l13 = tf.keras.layers.BatchNormalization(axis=1)(l12)
l14 = tf.keras.layers.MaxPooling2D((2, 2))(l13)
l15 = tf.keras.layers.Flatten()(l14)
l16 = tf.keras.layers.Dense(4, activation='softmax')(l15)

# features
conv_layer_index = [0, 3, 6, 9, 12]

# Generate feature output by predicting on the input image
layers = [l0, l3, l6, l9, l12]
test_fig = ix

for n in range(len(layers)):
    model_n = tf.keras.Model(inputs=[inputs], outputs=[layers[n]])
    model_n.set_weights(weights[0:(2 * (n + 1))])
    preds_test = model_n.predict(X_train, verbose=0)
    plt.figure(n, figsize=(10, 10))
    total_fig = np.shape(preds_test)[-1]
    cto = 1
    for i in range(0, total_fig):

        plt.subplot(int(np.ceil(np.sqrt(total_fig))), int(np.ceil(np.sqrt(total_fig))), cto)
        plt.imshow(preds_test[test_fig, :, :, i], cmap='gist_gray', vmin=0, vmax=np.max(preds_test[test_fig, :, :, i]))
        plt.axis('off')
        cto += 1
        if cto > total_fig:
            break

# filters
layer = 0
layer = layer * 2
max_value = np.max(weights[layer][:, :, :, :])
min_value = np.min(weights[layer][:, :, :, :])
for j in range(np.shape(weights[layer])[2]):
    plt.figure(j, figsize=(10, 10), edgecolor='black')
    total_fig = np.shape(weights[layer])[3]
    cto = 1
    if layer == 0:
        if j == 0:
            colormap = 'Reds'
            title = 'Red Channel Filters'
        elif j == 1:
            colormap = 'Greens'
            title = 'Green Channel Filters'
        else:
            colormap = 'Blues'
            title = 'Blue Channel Filters'
    else:
        colormap = 'Greys'
        title = 'Convolutional layer ' + str(layer) + ' char: ' + str(j+1)
    for i in range(0, total_fig):
        plt.subplot(int(np.ceil(np.sqrt(total_fig))), int(np.ceil(np.sqrt(total_fig))), cto)

        plt.imshow((weights[layer][:, :, j, i] - min_value) / (max_value - min_value), cmap=colormap, vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.suptitle(title, fontsize=32)
        cto += 1
        if cto > total_fig:
            break
