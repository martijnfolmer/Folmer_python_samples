import os
import shutil

from ultralytics import YOLO

"""
    Using the Ultralytics YOLO framework, train a classification model on the MNIST dataset, then save it as a .pt and
    .onnx file format for later use

    Ultralytics has some benefits over keras, the package does some pre-analysis of the training data, makes images you 
    can check and it generates result files with each run for easy comparison, and the exporting of the model to onnx 
    is build-in.
    
    The downside is that making your own model architecture is more difficult. But for most applications, yolo models
    will suffice.

    Mnist is a dataset of handwritten digits, and this model will classify an image with a written number as being of 
    a certain type (0-9)
    
    This will also download the mnist dataset if you don't already have it
    
    NOTE : this script in particular requires packages that are not needed in other scripts in this repository,
    namely ultralytics
    
    Author :        Martijn Folmer
    Date created :  23-02-2026
"""

class TrainMnist:
    def __init__(self, model_variant="yolo11n-cls.pt"):
        # Model / data parameters
        self.num_classes = 10           # The 10 numbers that mnist can be
        self.imgsz = 64                 # Minimum image size for YOLO classification models
        self.model_variant = model_variant

        # Load a pretrained YOLO classification model
        self.model = YOLO(self.model_variant)

        # Will be set after training
        self.results = None

    def train_model(self, batch_size=128, epochs=15):

        # Train the model on the MNIST dataset (auto-downloads if not present)
        self.results = self.model.train(
            data="mnist",
            epochs=epochs,
            imgsz=self.imgsz,
            batch=batch_size,
        )

        print("Training complete.")

    def validate_model(self):

        # Run validation on the test set
        metrics = self.model.val()

        print(f"Top-1 accuracy: {metrics.top1:.4f}")
        print(f"Top-5 accuracy: {metrics.top5:.4f}")

        return metrics

    def save_model(self, save_path_pt=None, save_path_onnx=None):

        if save_path_pt is not None:
            # Copy the best checkpoint to the desired location
            best_pt = self.model.trainer.best if self.model.trainer else None
            if best_pt and os.path.exists(best_pt):
                shutil.copy2(str(best_pt), save_path_pt)
                print(f"Model saved to: {save_path_pt}")
            else:
                # Fallback: save wherever the model currently points
                shutil.copy2(str(self.model.ckpt_path), save_path_pt)
                print(f"Model saved to: {save_path_pt}")

        if save_path_onnx is not None:
            # Export the model to ONNX format
            exported_path = self.model.export(format="onnx", imgsz=self.imgsz)
            shutil.copy2(str(exported_path), save_path_onnx)
            print(f"ONNX model saved to: {save_path_onnx}")


if __name__ == "__main__":
    pathToModels = "models"
    pathToPt = f"{pathToModels}/mnist_ultralytics.pt"
    pathToONNX = f"{pathToModels}/mnist_ultralytics.onnx"

    # make sure the target path where we store the models exists
    os.makedirs(pathToModels, exist_ok=True)

    # initialize the class
    TM = TrainMnist()

    # Train the model
    TM.train_model(epochs=20)

    # Validate the model
    TM.validate_model()

    # Export as pt and as onnx
    TM.save_model(save_path_pt=pathToPt, save_path_onnx=pathToONNX)

