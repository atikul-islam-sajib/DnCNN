import sys
import os
import argparse
import matplotlib.pyplot as plt
import imageio
import torch

sys.path.append("src/")

from config import (
    BEST_MODELS_PATH,
    PROCESSED_DATA_PATH,
    TRAIN_IMAGES_PATH,
    GIF_IMAGE_PATH,
    TEST_IMAGES_PATH,
)
from utils import device_init, load
from DnCNN import DnCNN
from vgg16 import VGG16


class Charts:
    """
    A class used to represent and perform operations for model visualization, including
    selecting the best model based on loss, loading data, normalizing images, creating GIFs from
    training images, and plotting comparison images (noisy, clean, predicted) using Matplotlib.

    Attributes:
        device (str): The device type (e.g., 'cpu', 'cuda', 'mps') for PyTorch operations.
        infinity (float): A high initial value for comparing and finding minimum loss.
        model (torch.nn.Module): The neural network model used for image denoising.
        best_model (torch.nn.Module): The best performing model based on loss.
        details (dict): A dictionary containing details of the best model and its loss.

    Methods:
        select_best_model(self):
            Selects the best model based on minimum loss from saved models.

        dataloader(self):
            Loads the dataloader from the processed data path.

        image_normalized(self, image):
            Normalizes an image tensor to be in the range [0, 1].

        create_gif(self):
            Creates a GIF from training images stored in a specified path.

        plot(self):
            Plots comparison images (noisy, clean, predicted) and saves a comparison figure and a training images GIF.
    """

    def __init__(self, device="mps", is_vgg16=False):
        """
        Initializes the Charts object with specified device for PyTorch operations and
        sets initial values for attributes.

        Parameters:
            device (str): The device type for PyTorch operations. Defaults to 'mps'.
        """
        self.device = device_init(device=device)
        self.is_vgg16 = is_vgg16
        self.infinity = float("inf")
        self.model = None
        self.best_model = None
        self.details = {"best_model": list(), "best_loss": list()}

    def select_best_model(self):
        """
        Selects the best model based on minimum loss from models saved in the BEST_MODELS_PATH.

        Returns:
            torch.nn.Module: The best performing model based on loss.

        Raises:
            Exception: If the BEST_MODELS_PATH does not exist.
        """
        if os.path.exists(BEST_MODELS_PATH):

            models = os.listdir(BEST_MODELS_PATH)
            for model in models:
                if (
                    self.infinity
                    > torch.load(os.path.join(BEST_MODELS_PATH, model))["loss"]
                ):
                    self.infinity = torch.load(os.path.join(BEST_MODELS_PATH, model))[
                        "loss"
                    ]
                    self.best_model = torch.load(os.path.join(BEST_MODELS_PATH, model))[
                        "model"
                    ]
            self.details["best_model"].append(self.best_model)
            self.details["best_loss"].append(self.infinity)

            return self.best_model
        else:
            raise Exception("Best model path is not found".title())

    def dataloader(self):
        """
        Loads and returns the dataloader object from a pickled file in the PROCESSED_DATA_PATH.

        Returns:
            DataLoader: The dataloader containing the dataset.

        Raises:
            Exception: If the PROCESSED_DATA_PATH does not exist or the dataloader file is not found.
        """
        if os.path.exists(PROCESSED_DATA_PATH):
            return load(os.path.join(PROCESSED_DATA_PATH, "dataloader.pkl"))
        else:
            raise Exception("Processed data path is not found".title())

    def image_normalized(self, image):
        """
        Normalizes an image tensor to have values in the range [0, 1].

        Parameters:
            image (torch.Tensor): The image tensor to be normalized.

        Returns:
            numpy.ndarray: The normalized image as a numpy array.
        """
        image = image.cpu().detach().numpy()
        return (image - image.min()) / (image.max() - image.min())

    def create_gif(self):
        """
        Creates and saves a GIF from images stored in TRAIN_IMAGES_PATH to GIF_IMAGE_PATH.

        Raises:
            Exception: If the TRAIN_IMAGES_PATH or GIF_IMAGE_PATH does not exist.
        """
        if os.path.exists(TRAIN_IMAGES_PATH) and os.path.exists(GIF_IMAGE_PATH):
            images = [
                imageio.imread(os.path.join(TRAIN_IMAGES_PATH, image))
                for image in os.listdir(TRAIN_IMAGES_PATH)
            ]
            imageio.mimsave(
                os.path.join(GIF_IMAGE_PATH, "train_masks.gif"), images, "GIF"
            )
        else:
            raise Exception("Train images path not found.".capitalize())

    def plot(self):
        """
        Plots comparison images for each set of noisy, clean, and predicted images in the test dataset.
        Saves a comparison figure to TEST_IMAGES_PATH and a GIF of training images to GIF_IMAGE_PATH.

        Raises:
            Exception: If the TEST_IMAGES_PATH does not exist.
        """
        try:
            if self.is_vgg16:
                self.model = VGG16().to(self.device)
            else:
                self.model = DnCNN().to(self.device)
        except Exception as e:
            print("The exception in the model is:", e)
        else:
            self.model.load_state_dict(self.select_best_model())

        finally:
            self.test_dataloader = self.dataloader()

            noise_images, clean_images = next(iter(self.test_dataloader))

            noise_images = noise_images.to(self.device)
            clean_images = clean_images.to(self.device)

            predicted_images = self.model(noise_images)

        plt.figure(figsize=(36, 24))

        for index, image in enumerate(predicted_images):
            noisy = noise_images[index].permute(1, 2, 0)
            noisy = self.image_normalized(noisy)

            plt.subplot(3 * 5, 3 * 8, 3 * index + 1)
            plt.imshow(noisy, cmap="gray")
            plt.title("Noisy")
            plt.axis("off")

            clean = clean_images[index].permute(1, 2, 0)
            clean = self.image_normalized(clean)

            plt.subplot(3 * 5, 3 * 8, 3 * index + 2)
            plt.imshow(clean, cmap="gray")
            plt.title("Clean")
            plt.axis("off")

            predicted = image.permute(1, 2, 0)
            predicted = self.image_normalized(predicted)

            plt.subplot(3 * 5, 3 * 8, 3 * index + 3)
            plt.imshow(predicted, cmap="gray")
            plt.title("Predict")
            plt.axis("off")

        plt.tight_layout()

        if os.path.exists(TEST_IMAGES_PATH):
            plt.savefig(os.path.join(TEST_IMAGES_PATH, "test.png"))

            try:
                self.create_gif()
            except Exception as e:
                print("The exception in the gif is:", e)
        else:
            raise Exception("Test images path is not found".title())

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define the Charts".title())
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device type for PyTorch operations".capitalize(),
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Define the arguments for testing".capitalize(),
    )

    args = parser.parse_args()

    if args.test:

        charts = Charts(device=args.device)
        charts.plot()

    else:
        raise Exception("Define the arguments for testing".title())
