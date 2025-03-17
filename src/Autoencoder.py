from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, BatchNormalization
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="MyLayers")
class Encoder(Model):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size

        self.my_layers = []
        for filter in filters:
            self.my_layers.append(
                Conv2D(filters=filter, kernel_size=kernel_size, strides=2, padding="same", activation='relu')
            )
            # self.my_layers.append(BatchNormalization(0.1))

            
    def call(self, inputs, training=False):
        for my_layer in self.my_layers:
            inputs = my_layer(inputs)

        return inputs

    def get_config(self):
        config = super().get_config()  # Get the default config from the parent class
        config.update({
            'filters': self.filters,  # Include the filters used to initialize
            'kernel_size': self.kernel_size,  # Include the kernel_size used to initialize
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Create the instance of Encoder using the config
        return cls(filters=config['filters'], kernel_size=config['kernel_size'])


@keras.saving.register_keras_serializable(package="MyLayers")
class Decoder(Model):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size

        self.my_layers = []
        for filter in filters:
            self.my_layers.append(
                Conv2DTranspose(filters=filter, kernel_size=kernel_size, strides=2, padding="same", activation='relu')
            )
            # self.my_layers.append(BatchNormalization(0.1))

        self.my_layers.append(
            Conv2D(filters=3, kernel_size=3, strides=1, padding="same", activation='sigmoid')
        )

    
    def call(self, inputs, training=False):
        for my_layer in self.my_layers:
            inputs = my_layer(inputs)        
        return inputs
    
    def get_config(self):
        config = super().get_config()  # Get the default config from the parent class
        config.update({
            'filters': self.filters,  # Include the filters used to initialize
            'kernel_size': self.kernel_size,  # Include the kernel_size used to initialize
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Create the instance of Decoder using the config
        return cls(filters=config['filters'], kernel_size=config['kernel_size'])

    
@keras.saving.register_keras_serializable(package="MyLayers")
class Autoencoder(Model):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.encoder = Encoder(filters=filters, kernel_size=kernel_size)
        self.decoder = Decoder(filters=filters[::-1], kernel_size=kernel_size) # orginal_img_size=orginal_img_size

    def call(self, inputs, training=False):
        # orginal_img_size = inputs.shape[1:-1]
        inputs = self.encoder(inputs)
        inputs = self.decoder(inputs)
        # inputs = tf.image.resize(inputs, orginal_img_size)
        return inputs

    def get_config(self):
        config = super().get_config()  # Get the default config from the parent class
        config.update({
            'filters': self.filters,  # Include the filters used to initialize
            'kernel_size': self.kernel_size,  # Include the kernel_size used to initialize
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Create the instance of Autoencoder using the config
        return cls(filters=config['filters'], kernel_size=config['kernel_size'])
