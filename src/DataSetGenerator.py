import os 
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
from sklearn.model_selection import train_test_split


class DataSetGenerator(Sequence):
    def __init__(self, batch_size, root_file, mode="train", test_size=0.2):
        super().__init__()

        self.batch_size = batch_size
        self.root_file = root_file
        self.list_of_all_img_paths = self.get_list_of_files()
        
        train_paths, test_paths = train_test_split(self.list_of_all_img_paths, test_size=test_size, random_state=42)

        self.list_of_img_paths = train_paths if mode == 'train' else test_paths
        self.indexes = np.arange(len(self.list_of_img_paths))


    def get_list_of_files(self):
        img_path = []
        for root, _, files in os.walk(self.root_file):
            for file in files:
                img_path.append(os.path.join(root, file))

        # import random
        # random_20_percent = random.sample(img_path, int(len(img_path) * 0.2))
        return img_path


    def __len__(self):
        return len(self.list_of_img_paths) // self.batch_size


    def __getitem__(self, idx):
        batch_indexes = self.list_of_img_paths[idx*self.batch_size : (idx+1)*self.batch_size]

        x_batch = []
        y_batch = []
        
        for i, image_path in enumerate(batch_indexes):

            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [32, 32])
            image = image / 255.0
            image_gray = tf.image.rgb_to_grayscale(image)
            x_batch.append(image_gray)
            y_batch.append(image)

        return tf.stack(x_batch), tf.stack(y_batch)
    

    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        np.random.shuffle(self.indexes)
