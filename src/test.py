import sys
import os
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


class Charts:
    def __init__(self, device="mps"):
        self.device = device_init(device=device)
        self.infinity = float("inf")
        self.model = None
        self.best_model = None
        self.details = {"best_model": list(), "best_loss": list()}

    def select_best_model(self):
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
        if os.path.exists(PROCESSED_DATA_PATH):
            return load(os.path.join(PROCESSED_DATA_PATH, "dataloader.pkl"))
        else:
            raise Exception("Processed data path is not found".title())

    def image_normalized(self, image):
        image = image.cpu().detach().numpy()
        return (image - image.min()) / (image.max() - image.min())

    def create_gif(self):
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
        try:
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
    charts = Charts()
    charts.plot()
