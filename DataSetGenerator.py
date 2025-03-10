import os 
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np

class DataSetGenerator(Sequence):
    def __init__(self, batch_size, root_file):
        

        self.batch_size = batch_size
        self.root_file = root_file
        self.list_of_img_paths = self.get_list_of_files()
        self.indexes = np.arange(len(self.list_of_img_paths))


    def get_list_of_files(self):
        img_path = []
        for root, _, files in os.walk(self.root_file):
            for file in files:
                img_path.append(os.path.join(root, file))
        return img_path


    def __len__(self):
        return len(self.list_of_img_paths) // self.batch_size


    def __getitem__(self, idx):
        image_path = self.list_of_img_paths[idx]

        image = cv2.imread(image_path) #BGR format
        image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #RGB format

        image_grey = cv2.resize(image_grey, (300, 450))
        image = cv2.resize(image, (300, 450))

        image_grey = image_grey / 255.0
        image = image / 255.0

        image_grey_tensor = tf.convert_to_tensor(image_grey, dtype=tf.float32)
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        image_grey_tensor = tf.expand_dims(image_grey_tensor, axis=-1)  # (H, W, 1)
        x = tf.expand_dims(image_grey_tensor, axis=0)
        y = tf.expand_dims(image, axis=0)
        
        return x, y