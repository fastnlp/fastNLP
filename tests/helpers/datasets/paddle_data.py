import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle.io import Dataset
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Dataset


class PaddleNormalDataset(Dataset):
    def __init__(self, num_of_data=1000):
        self.num_of_data = num_of_data
        self._data = list(range(num_of_data))

    def __len__(self):
        return self.num_of_data

    def __getitem__(self, item):
        return self._data[item]


class PaddleRandomMaxDataset(Dataset):
    def __init__(self, num_samples, num_features):
        self.x = paddle.randn((num_samples, num_features))
        self.y = self.x.argmax(axis=-1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return {"x": self.x[item], "y": self.y[item]}
