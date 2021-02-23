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
from skimage.transform import resize
from tqdm import tqdm

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

# Test dataset info
TEST_PATH_IMAGES = BASE_PATH + separator_dir + 'dataset' + separator_dir + 'test'

# Get augmented images ids
Test_images_files = next(os.walk(TEST_PATH_IMAGES))[2]

X_test = np.zeros((len(Test_images_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

# Image resizing process
print('Resizing test images...')
for n, id_ in tqdm(enumerate(Test_images_files), total=len(Test_images_files)):
    path_image = TEST_PATH_IMAGES + '\\'
    img = imread(path_image + Test_images_files[n])[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    
# Load model
model_name = os.listdir('models')
model = tf.keras.models.load_model('models' + separator_dir + model_name[-1])

# Make a prediction
preds_test = model.predict(X_test, verbose=1)
preds_test_t = (preds_test > 0.8).astype(np.uint8)

# Perform a sanity check on some random validation samples
ix = 5
imshow(X_test[ix])
plt.xticks([])
plt.yticks([])
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.xticks([])
plt.yticks([])
plt.show()

# Deconvolution process
weights = model.get_weights()

# Input layer
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

# Contraction path
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

# Expansive path
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.2)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

layers = [c1, c2, c3, c4, c7, c8, c9]
test_fig = 5

for n in range(len(layers)):
    model_n = tf.keras.Model(inputs=[inputs], outputs=[layers[n]])
    model_n.set_weights(weights[0:(2*(n+1))])
    preds_test = model_n.predict(X_test, verbose=0)
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

plt.figure()
imagerr = preds_test_t*X_test
imshow(imagerr[test_fig])
plt.xticks([])
plt.yticks([])
plt.show()

layer = 0
    
layer = layer*2    
max_value = np.max(weights[layer][:, :, :, :])
min_value = np.min(weights[layer][:, :, :, :])
print(max_value)
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
        plt.imshow((weights[layer][:, :, j, i]-min_value)/(max_value-min_value), cmap=colormap, vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.suptitle(title, fontsize=32)
        cto += 1
        if cto > total_fig:
            break
