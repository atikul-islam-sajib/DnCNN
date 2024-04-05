import sys
import argparse
import os
import zipfile
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

sys.path.append("src/")

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, FILE_PATH
from utils import clean, dump, load


class Loader(Dataset):
    """
    Loader class to handle the loading, preprocessing, and splitting of image datasets for training and testing.

    Attributes:
        image_path (str): Path to the zip file containing the dataset.
        batch_size (int): Number of images per batch.
        image_size (int): The height and width to which the images will be resized.
        split_ratio (float): The ratio of the dataset to be used for testing.

    Methods:
        unzip_folder: Extracts the dataset from a zip file to a specified path.
        image_transforms: Returns a composition of image transformations.
        image_split: Splits the dataset into training and testing sets.
        create_dataloader: Creates and saves data loaders for the dataset.
        dataset_details: Prints details about the dataset.
        show_images: Displays a set of noisy and clean images for visualization.

    Note:
        Ensure that the paths for RAW_DATA_PATH and PROCESSED_DATA_PATH are correctly set in the 'config' module.

    Example:
        >>> loader = Loader(image_path="./data/data.zip", batch_size=16, image_size=64, split_ratio=0.25)
        >>> loader.unzip_folder()
        >>> dataloader = loader.create_dataloader()
        >>> loader.dataset_details()
        >>> loader.show_images()
    """

    def __init__(self, image_path=None, batch_size=16, image_size=64, split_ratio=0.25):
        """
        Initializes the Loader with dataset path, batch size, image size, and split ratio.

        Parameters:
            image_path (str): Path to the dataset zip file.
            batch_size (int): Number of images in each batch of data.
            image_size (int): Size (height and width) to which images are resized.
            split_ratio (float): Fraction of the dataset to allocate to the test set.

        """
        self.image_path = image_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.split_ratio = split_ratio

        self.clean_images = list()
        self.noise_images = list()

    def unzip_folder(self):
        """
        Extracts images from a zipped file to RAW_DATA_PATH.

        Raises:
            Exception: If RAW_DATA_PATH does not exist or the zip file is invalid.
        """

        try:
            clean()
            pass
        except Exception as e:
            print("The exception is: {}".format(e))

        if os.path.exists(RAW_DATA_PATH):
            with zipfile.ZipFile(self.image_path, "r") as zip_ref:
                zip_ref.extractall(
                    os.path.join(
                        RAW_DATA_PATH,
                    )
                )
        else:
            raise Exception("Please provide a valid path".capitalize())

    def image_transforms(self):
        """
        Defines and returns the transformations to be applied to the images.

        Returns:
            torchvision.transforms.Compose: A composed transform with resizing, tensor conversion, and normalization.

        """
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def image_split(self, **kwargs):
        """
        Splits the dataset into training and testing sets based on `split_ratio`.

        Parameters:
            **kwargs: Expects 'noise_images' and 'clean_images' lists.

        Returns:
            tuple: Contains split datasets (train_noise, test_noise, train_clean, test_clean).

        Raises:
            ValueError: If `kwargs` does not contain 'noise_images' or 'clean_images'.

        """
        return train_test_split(
            kwargs["noise_images"],
            kwargs["clean_images"],
            test_size=self.split_ratio,
            random_state=42,
        )

    def create_dataloader(self):
        """
        Creates and saves DataLoader objects for training and testing datasets.

        Returns:
            DataLoader: The combined DataLoader for the entire dataset.

        Raises:
            Exception: If necessary paths do not exist or dataset is not properly formatted.
        """
        if os.path.exists(RAW_DATA_PATH):
            dataset = os.listdir(RAW_DATA_PATH)[0]

            clean_images = os.path.join(RAW_DATA_PATH, dataset, "clean_images")
            noise_images = os.path.join(RAW_DATA_PATH, dataset, "noisy_images")

            for image in os.listdir(clean_images):
                clean_image_path = os.path.join(clean_images, image)

                if image in os.listdir(noise_images):
                    noise_image_path = os.path.join(noise_images, image)
                else:
                    continue

                self.clean_images.append(
                    self.image_transforms()(
                        Image.fromarray(cv2.imread(clean_image_path))
                    )
                )

                self.noise_images.append(
                    self.image_transforms()(
                        Image.fromarray(cv2.imread(noise_image_path))
                    )
                )

            try:
                image_split = self.image_split(
                    clean_images=self.clean_images, noise_images=self.noise_images
                )

                dataloader = DataLoader(
                    dataset=list(zip(self.noise_images, self.clean_images)),
                    batch_size=self.batch_size * 6,
                    shuffle=True,
                )

                train_dataloader = DataLoader(
                    dataset=list(zip(image_split[0], image_split[2])),
                    batch_size=self.batch_size,
                    shuffle=True,
                )

                test_dataloader = DataLoader(
                    dataset=list(zip(image_split[1], image_split[3])),
                    batch_size=self.batch_size,
                    shuffle=True,
                )

                if os.path.exists(PROCESSED_DATA_PATH):

                    dump(
                        value=dataloader,
                        filename=os.path.join(PROCESSED_DATA_PATH, "dataloader.pkl"),
                    )

                    dump(
                        value=train_dataloader,
                        filename=os.path.join(
                            PROCESSED_DATA_PATH, "train_dataloader.pkl"
                        ),
                    )

                    dump(
                        value=test_dataloader,
                        filename=os.path.join(
                            PROCESSED_DATA_PATH, "test_dataloader.pkl"
                        ),
                    )
                else:
                    raise Exception("Please provide a valid path".capitalize())

            except Exception as e:
                print("The exception is: {}".format(e))

            return dataloader

    @staticmethod
    def dataset_details():
        """
        Prints the total number of images and the shape of clean and noisy images in the dataset.

        Raises:
            Exception: If PROCESSED_DATA_PATH does not exist.
        """
        if os.path.exists(PROCESSED_DATA_PATH):
            dataloader = load(
                filename=os.path.join(PROCESSED_DATA_PATH, "dataloader.pkl")
            )

            clean, noise = next(iter(dataloader))

            total_data = sum(clean.size(0) for clean, _ in dataloader)

            print("Total number of images: {}".format(total_data))
            print("Clean images shape : {}".format(clean.size()))
            print("Noisy images shape : {}".format(noise.size()))

        else:
            raise Exception("Please provide a valid path".capitalize())

    @staticmethod
    def show_images():
        """
        Visualizes a sample of noisy and clean images from the dataset.

        Raises:
            Exception: If PROCESSED_DATA_PATH does not exist or image files are missing.
        """
        if os.path.exists(PROCESSED_DATA_PATH):
            dataloader = load(
                filename=os.path.join(PROCESSED_DATA_PATH, "dataloader.pkl")
            )
        else:
            raise Exception("Please provide a valid path".capitalize())

        plt.figure(figsize=(40, 15))

        noise, clean = next(iter(dataloader))

        for index, image in enumerate(noise):
            noise_image = image.permute(1, 2, 0)
            noise_image = (noise_image - noise_image.min()) / (
                noise_image.max() - noise_image.min()
            )

            clean_image = clean[index].permute(1, 2, 0)
            clean_image = (clean_image - clean_image.min()) / (
                clean_image.max() - clean_image.min()
            )

            plt.subplot(2 * 4, 2 * 6, 2 * index + 1)
            plt.imshow(noise_image)
            plt.title("Noisy")
            plt.axis("off")

            plt.subplot(2 * 4, 2 * 6, 2 * index + 2)
            plt.imshow(clean_image)
            plt.title("Clean")
            plt.axis("off")

        plt.tight_layout()
        if os.path.exists(FILE_PATH):
            plt.savefig(os.path.join(FILE_PATH, "raw_image.png"))
        else:
            raise Exception("Please provide a valid path".capitalize())

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the Dataloader for DnCNN".title()
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default="./data/data.zip",
        help="Path to the zip file containing the dataset".capitalize(),
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, default=64, help="Image size".capitalize()
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.2,
        help="Spit the dataset".capitalize(),
    )

    args = parser.parse_args()

    if args.image_path:
        loader = Loader(
            image_path=args.image_path,
            batch_size=args.batch_size,
            image_size=args.image_size,
            split_ratio=args.split_ratio,
        )

        loader.unzip_folder()
        dataloader = loader.create_dataloader()

        try:
            loader.dataset_details()

            loader.show_images()

        except Exception as e:
            print("The exception is: {}".format(e))

    else:
        raise Exception("Please provide a valid path".capitalize())
