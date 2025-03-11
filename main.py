from Autoencoder import Autoencoder
from DataSetGenerator import DataSetGenerator
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import kagglehub
import shutil
import os
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import click
import tensorflow as tf 



def preper_dataset():
    path = kagglehub.dataset_download("sautkin/imagenet1k0")
    shutil.copytree(path, "./2", dirs_exist_ok=True)

def setup_callbacks(model_file_name):
    log_dir = f"logs/{model_file_name}"
    checkpoint_dir = "checkpoints"

    tensorboard_callback = TensorBoard(log_dir=log_dir)

    checkpoint_callback = ModelCheckpoint(
        os.path.join(checkpoint_dir, f"{model_file_name}.keras"),
        monitor='loss',
        save_best_only=True, 
        mode='min',
        verbose=1
    )

    return [tensorboard_callback, checkpoint_callback]

@click.command()
@click.option('--number_of_epoches', default=10, type=int, help='number_of_epoches')
@click.option('--batch_size', default=8, type=int, help='batch_size')
@click.option('--initial_learning_rate', default=1e-3, type=float, help='initial_learning_rate')
@click.option('--filters', default=[128, 64, 32, 16], type=int, multiple=True, help='filters')
@click.option('--kernel_size', default=(3, 3), type=(int, int), help='kernel_size')
def main(number_of_epoches, batch_size, initial_learning_rate, filters, kernel_size):

    tainDataSetGenerator = DataSetGenerator(batch_size=batch_size, root_file="2", mode="train")
    testDataSetGenerator = DataSetGenerator(batch_size=batch_size, root_file="2", mode="test")

    steps_per_epoch = (len(tainDataSetGenerator.list_of_img_paths) // batch_size) 
    lr_schedule = PiecewiseConstantDecay(
        boundaries=[steps_per_epoch * (number_of_epoches-2),
                    steps_per_epoch * (number_of_epoches-1)],
        values=[initial_learning_rate, initial_learning_rate/2, initial_learning_rate/5]
    )
    optimizer = AdamW(learning_rate=lr_schedule)

    model = Autoencoder(filters=filters, kernel_size=kernel_size)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    
    model_file_name = f"best_model_number_of_epoches:{number_of_epoches}_batch_size:{batch_size}_initial_learning_rate:{initial_learning_rate}_filters:{filters}_kernel_size:{kernel_size}"
    callbacks = setup_callbacks(model_file_name)

    model.fit(tainDataSetGenerator, epochs=number_of_epoches, callbacks=callbacks, validation_data=testDataSetGenerator)

if __name__ == "__main__":
    main()
