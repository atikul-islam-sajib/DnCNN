import sys
import os
import unittest
import torch

sys.path.append("src/")

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from utils import dump, load
from DnCNN import DnCNN


class UnitTest(unittest.TestCase):
    """
    A class for unit testing the DataLoader objects used in a deep learning project.
    """

    def setUp(self):
        if os.path.exists(PROCESSED_DATA_PATH):
            self.dataloader = load(
                filename=os.path.join(PROCESSED_DATA_PATH, "dataloader.pkl")
            )
            self.train_dataloader = load(
                filename=os.path.join(PROCESSED_DATA_PATH, "train_dataloader.pkl")
            )
            self.test_dataloader = load(
                filename=os.path.join(PROCESSED_DATA_PATH, "test_dataloader.pkl")
            )

            self.model = DnCNN()

        else:
            raise Exception("Data not found".capitalize())

        self.total_data = 100
        self.total_params = 558336

    def tearDown(self):
        """
        Teardown method to clean up after each test.
        """
        self.total_data = None

    def test_total_datasets(self):
        """
        Test to ensure the total dataset size matches the expected size.
        """
        self.assertEquals(
            sum(noise.size(0) for noise, clean in self.dataloader), self.total_data
        )

    def test_size_dataset(self):
        """
        Test to verify that the size of the batches in the dataset is as expected.
        """
        noise, _ = next(iter(self.dataloader))

        self.assertEquals(noise.size(), torch.Size([24, 3, 64, 64]))

    def test_DnCNN_model_params(self):
        """
        Test to verify that the params of the in the model is as expected.
        """
        self.assertEqual(
            sum(params.numel() for params in self.model.parameters()), self.total_params
        )

    def test_DnCNN_model(self):
        """
        Test to verify that the size of the output in the model is as expected.
        """
        data = torch.randn(1, 3, 64, 64)
        self.assertEqual(self.model(data).size(), torch.Size([1, 3, 64, 64]))


if __name__ == "__main__":
    unittest.main()
