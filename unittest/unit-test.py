import sys
import os
import unittest
import torch

sys.path.append("src/")

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from utils import dump, load


class UnitTest(unittest.TestCase):

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

        else:
            raise Exception("Data not found".capitalize())

        self.total_data = 100

    def tearDown(self):
        self.total_data = None

    def test_total_datasets(self):
        self.assertEquals(
            sum(noise.size(0) for noise, clean in self.dataloader), self.total_data
        )

    def test_size_dataset(self):
        noise, _ = next(iter(self.dataloader))

        self.assertEquals(noise.size(), torch.Size([24, 3, 64, 64]))


if __name__ == "__main__":
    unittest.main()
