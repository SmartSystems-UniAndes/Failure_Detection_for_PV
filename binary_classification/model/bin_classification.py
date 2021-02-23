# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 15:37:26 2020
@author: ANDRES FELIPE FLOREZ
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from skimage.io import imread, imshow
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

# Dataset info
TRAIN_PATH_IMAGES = BASE_PATH + separator_dir + 'augmented_train_data' + separator_dir + 'images'
TRAIN_PATH_MASKS = BASE_PATH + separator_dir + 'augmented_train_data' + separator_dir + 'masks'

# Get augmented images ids
Train_images_files = next(os.walk(TRAIN_PATH_IMAGES))[2]
Train_masks_files = next(os.walk(TRAIN_PATH_MASKS))[2]

X_train = np.zeros((len(Train_images_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(Train_images_files), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
Y_train_class = np.zeros((len(Train_images_files), 1), dtype=np.bool)

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
        Y_train_class[n] = 1
    if name == 'N_C_T':
        Y_train_class[n] = 1
    if name == 'N_S_S':
        Y_train_class[n] = 0

# Randomize order
index = np.arange(X_train.shape[0])
np.random.shuffle(index)

X_train = X_train[index]
Y_train_class = Y_train_class[index]

# Model
model = tf.keras.models.Sequential()
model.add(Conv2D(16, (5, 5), input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_pv.h5', verbose=1, save_best_only=True)

callbacks = [tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss'),
             tf.keras.callbacks.TensorBoard(log_dir='logs')]

# Train
results = model.fit(X_train, Y_train_class, validation_split=0.1, batch_size=10, epochs=20, callbacks=callbacks)

# Save models
Val_acc = results.history['val_acc'][-1]
model.save('models' + separator_dir + 'bin_class_val_acc_' + str(round(Val_acc, 4)) + '.h5')

# Plot historic
plt.figure(1, figsize=(5, 5))
plt.plot(results.history['loss'], label='Train')
plt.plot(results.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

plt.figure(2, figsize=(5, 5))
plt.plot(results.history['acc'], label='Train acc')
plt.plot(results.history['val_acc'], label='Validation acc')
plt.legend()
plt.show()

# Get some results
preds_train = model.predict(X_train, verbose=1)
preds_train = (preds_train > 0.5).astype(np.uint8)

ix = random.randint(0, len(preds_train) - 1)
imshow(X_train[ix])
plt.show()

print('Prediction: ', preds_train[ix])
print('True: ', Y_train_class[ix])
