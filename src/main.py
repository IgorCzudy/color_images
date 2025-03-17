from src.Autoencoder import Autoencoder
from src.DataSetGenerator import DataSetGenerator
from tensorflow.keras.optimizers import AdamW, Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import kagglehub
import shutil
import os
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import click
import tensorflow as tf 
import keras 
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16
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


@keras.saving.register_keras_serializable()  # Register the loss function
def ssim_loss(y_true, y_pred):
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    # ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)
    return mse_loss


@click.command()
@click.option('--number_of_epoches', default=15, type=int, help='number_of_epoches')
@click.option('--batch_size', default=128, type=int, help='batch_size')
@click.option('--initial_learning_rate', default=5e-3, type=float, help='initial_learning_rate')
@click.option('--filters', default=[64, 128, 256], type=int, multiple=True, help='filters') #128, 64, 32, 16
@click.option('--kernel_size', default=3, type=int, help='kernel_size')
def main(number_of_epoches, batch_size, initial_learning_rate, filters, kernel_size):
    preper_dataset()

    tainDataSetGenerator = DataSetGenerator(batch_size=batch_size, root_file="2", mode="train")
    testDataSetGenerator = DataSetGenerator(batch_size=batch_size, root_file="2", mode="test")

    initial_learning_rate = 1e-3
    final_learning_rate = 1e-5
    train_size = len(tainDataSetGenerator.list_of_img_paths)
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/number_of_epoches)
    steps_per_epoch = int(train_size/batch_size)


    # Define the scheduler — Exponential Decay as an example
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor,
        staircase=True  # If True, decay in discrete intervals
    )

    # Pass the scheduler into the AdamW optimizer
    optimizer = Adam(learning_rate=lr_scheduler)

    model = Autoencoder(filters=filters, kernel_size=kernel_size)

    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    
    model_file_name = f"2"
    callbacks = setup_callbacks(model_file_name)

    model.fit(tainDataSetGenerator, epochs=number_of_epoches, callbacks=callbacks, validation_data=testDataSetGenerator)

if __name__ == "__main__":
    main()
