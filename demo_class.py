import os
import numpy as np

from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Image Parameters
IMG_WIDTH = 200
IMG_HEIGHT = 200
IMG_CHANNELS = 3


class RunDemo:
    def __init__(self,
                 images,
                 segmentation_model,
                 bin_classification_model,
                 quat_classification_model,
                 output_path
                 ):
        self.images_files = images
        self.output_path = output_path

        # Load Models
        self.segmentation_model = tf.keras.models.load_model(segmentation_model)
        self.bin_classification_model = tf.keras.models.load_model(bin_classification_model)
        self.quat_classification_model = tf.keras.models.load_model(quat_classification_model)

        # Load Dataset
        self.dataset = self.__read_images()

    def __read_images(self):
        print("Loading Images...")
        data_images = []
        for image_files in tqdm(self.images_files):
            # Read image only as RGB
            im = imread(image_files)[:, :, :IMG_CHANNELS]
            # Resize
            im = resize(im, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

            data_images.append(im)

        data_images = np.stack(data_images)

        return data_images

    def run_segmentation(self):
        print("|-------------------------------------------------------------------|")
        print("Segmenting images...")
        seg_images = self.segmentation_model.predict(self.dataset, verbose=1)

        for i in range(len(seg_images)):
            im_name = os.path.basename(self.images_files[i]).split(".")[0]
            path = os.path.join(self.output_path, f"{im_name}.jpg")
            im = tf.keras.preprocessing.image.array_to_img(seg_images[i])
            im.save(path)

        print(f"Segmented images saved at {os.path.abspath(self.output_path)}")
        print("|-------------------------------------------------------------------|")

        return seg_images

    def run_bin_classification(self):
        print("|-------------------------------------------------------------------|")
        print("Classifying images...")
        classification_results = self.bin_classification_model.predict(self.dataset, verbose=1)
        classification_results = (classification_results > 0.5).astype(np.uint8)

        print("Results for binary classification:")
        for i in range(len(self.images_files)):
            im = os.path.basename(self.images_files[i]).split(".")[0]
            if classification_results[i] == 1:
                with_fail = "with fault."
            else:
                with_fail = "without fault."

            print(f"- Image {im} {with_fail}")
        print("|-------------------------------------------------------------------|")

        return classification_results

    def run_quat_classification(self):
        print("|-------------------------------------------------------------------|")
        print("Classifying images...")
        classification_results = self.quat_classification_model.predict(self.dataset, verbose=1)
        classification_results = (classification_results > 0.5).astype(np.uint8)

        print("Results for quaternary classification:")
        for i in range(len(self.images_files)):
            im = os.path.basename(self.images_files[i]).split(".")[0]
            clf_result = np.argmax(classification_results[i])
            with_fail = None
            if clf_result == 0:
                with_fail = "without fault."
            elif clf_result == 1:
                with_fail = "with cracks."
            elif clf_result == 2:
                with_fail = "with shadow."
            elif clf_result == 3:
                with_fail = "with dust."
            else:
                Exception(f"{clf_result} is an incorrect prediction")

            print(f"- Image {im} {with_fail}")

        print("|-------------------------------------------------------------------|")

        return classification_results

