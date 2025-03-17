from PIL import Image
import numpy as np
import tensorflow as tf 
from keras.models import load_model
from .Autoencoder import Autoencoder, Encoder, Decoder
from pathlib import Path

__version__="0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

loaded_model = load_model(f"{BASE_DIR}/finalResult.keras",
                          custom_objects={
        'Encoder': Encoder,
        'Decoder': Decoder,
        'Autoencoder': Autoencoder,
    }
)



def predict_pipeline(contents):
    
    image_gray = tf.image.decode_jpeg(contents, channels=1)
    image_gray = tf.cast(image_gray, tf.float32)
    image_gray = image_gray / 255.0
    image_gray = tf.expand_dims(image_gray, axis=0) # add batch size dim 
    
    image_predicted = loaded_model(image_gray)[0]

    rgb_array = image_predicted.numpy()
    rgb_array = np.clip(rgb_array * 255, 0, 255).astype(np.uint8)

    image_pil = Image.fromarray(rgb_array)
    return image_pil
    # buffer = io.BytesIO()

    # image_pil.save(buffer, format="JPEG")

    # buffer.seek(0)
    # return buffer
