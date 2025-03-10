from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose


class Encoder(Model):
    def __init__(self, filters, kernel_size):
        super().__init__()

        self.my_layers = []
        for filter in filters:
            self.my_layers.append(
                Conv2D(filters=filter, kernel_size=kernel_size, strides=(2,2), padding="same", activation='relu')
            )
            
    def call(self, inputs, training=False):
        for my_layer in self.my_layers:
            inputs = my_layer(inputs)

        return inputs
    
class Decoder(Model):
    def __init__(self, filters, kernel_size):
        super().__init__()

        self.my_layers = []
        for filter in filters:
            self.my_layers.append(
                Conv2DTranspose(filters=filter, kernel_size=kernel_size, strides=2, padding="same", activation='relu')
            )
        self.my_layers.append(
            Conv2D(filters=3, kernel_size=(3, 3), strides=1, padding="same", activation='sigmoid')
        )

    
    def call(self, inputs, training=False):
        for my_layer in self.my_layers:
            inputs = my_layer(inputs)        
        return inputs
    

class Autoencoder(Model):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.encoder = Encoder(filters=filters, kernel_size=kernel_size)
        self.decoder = Decoder(filters=filters[::-1], kernel_size=kernel_size) # orginal_img_size=orginal_img_size

    def call(self, inputs, training=False):
        orginal_img_size = inputs.shape[1:-1]
        inputs = self.encoder(inputs)
        inputs = self.decoder(inputs)
        inputs = tf.image.resize(inputs, orginal_img_size)
        return inputs
