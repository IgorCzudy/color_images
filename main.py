from Autoencoder import Autoencoder
from DataSetGenerator import DataSetGenerator
from tensorflow.keras.optimizers import AdamW
import kagglehub
import shutil

def preper_dataset():
    path = kagglehub.dataset_download("sautkin/imagenet1k0")
    shutil.copytree(path, ".")




def main():
    dataSetGenerator = DataSetGenerator(batch_size=16, root_file="2")
    optimizer = AdamW()
    model = Autoencoder(filters=[128, 64, 32, 16], kernel_size=(3, 3))
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    model.fit(dataSetGenerator, epochs=50)



if __name__ == "__main__":
    main()