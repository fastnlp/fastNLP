import paddle
from paddle.io import Dataset
import numpy as np


class PaddleNormalDataset(Dataset):
    def __init__(self, num_of_data=1000):
        self.num_of_data = num_of_data
        self._data = list(range(num_of_data))

    def __len__(self):
        return self.num_of_data

    def __getitem__(self, item):
        return self._data[item]


class PaddleRandomDataset(Dataset):
    def __init__(self, num_of_data=1000, features=64, labels=10):
        self.num_of_data = num_of_data
        self.x = [
            paddle.rand((features,))
            for i in range(num_of_data)
        ]
        self.y = [
            paddle.rand((labels,))
            for i in range(num_of_data)
        ]

    def __len__(self):
        return self.num_of_data

    def __getitem__(self, item):
        return {"x": self.x[item], "y": self.y[item]}


class PaddleDataset_MNIST(Dataset):
    def __init__(self, mode="train"):

        self.dataset = [
            (
                np.array(img).astype('float32').reshape(-1),
                label
            ) for img, label in paddle.vision.datasets.MNIST(mode=mode)
        ]

    def __getitem__(self, idx):
        return {"x": self.dataset[idx][0], "y": self.dataset[idx][1]}

    def __len__(self):
        return len(self.dataset)



