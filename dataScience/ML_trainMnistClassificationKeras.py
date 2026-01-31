import os
import subprocess
import sys
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

"""
    Using the Keras backend, train a simple mnist classification model, then save it as an .h5 and .onnx file format for
    later use
    
    Mnist is a dataset of handwritten digits, and this model will classify an image with a written number as being of 
    a certain type (0-9)
    
    This will also download the mnist dataset if you don't already have it
    
    NOTE : this script in particular requires packages that are not needed in other scripts in this repository,
    namely tensorflow and tf2onnx and onnx
    
    Author :        Martijn Folmer
    Date created :  31-01-2026
"""


class TrainMnist:
    def __init__(self):
        # Model / data parameters
        self.num_classes = 10           # The 10 numbers that mnist can be
        self.input_shape = (28, 28, 1)  # input size of the mnist data

        # load it from the tf.keras.datasets and prepare for training
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_mnist_data()

        # Build the model we train
        self.model = self.build_model()

    def load_mnist_data(self):
        # Load the data and split it between train and test sets
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # convert class vectors to binary class matrices
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)

        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")

        return x_train, y_train, x_test, y_test

    def build_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.summary()
        return model

    def train_model(self, batch_size=128, epochs=15):

        # compile and train the model
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def save_model(self, save_path_keras=None, save_path_onnx=None):

        if save_path_keras is not None:
            self.model.save(save_path_keras)

        if save_path_onnx is not None:
            # Keras 3 / TF 2.16+: export SavedModel for inference
            with tempfile.TemporaryDirectory() as tmpdir:
                saved_model_dir = os.path.join(tmpdir, "saved_model")
                # Keras 3 has model.export(); if you don't have it, see note below.
                self.model.export(saved_model_dir)

                # Convert SavedModel -> ONNX via tf2onnx CLI
                subprocess.check_call([
                    sys.executable, "-m", "tf2onnx.convert",
                    "--saved-model", saved_model_dir,
                    "--output", save_path_onnx,
                    "--opset", "18",
                ])


if __name__ == "__main__":
    pathToModels = "models"
    pathToKeras = f"{pathToModels}/mnist.keras"
    pathToONNX = f"{pathToModels}/mnist.onnx"

    # make sure the target path where we store the models exists
    os.makedirs(pathToModels, exist_ok=True)

    # initialize the class
    TM = TrainMnist()

    # Train the model
    TM.train_model(epochs=20)

    # Export as h5 and as onnx
    TM.save_model(save_path_keras=pathToKeras, save_path_onnx=pathToONNX)
