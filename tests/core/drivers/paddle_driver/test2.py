import torch
# from torch.utils.data import DataLoader, Dataset
import paddle
from paddle.io import Dataset, DataLoader
paddle.device.set_device("cpu")
class NormalDataset(Dataset):
    def __init__(self, num_of_data=1000):
        self.num_of_data = num_of_data
        self._data = list(range(num_of_data))

    def __len__(self):
        return self.num_of_data

    def __getitem__(self, item):
        return self._data[item]
dataset = NormalDataset(20)
dataloader = DataLoader(dataset, batch_size=2, use_buffer_reader=False)
for i, b in enumerate(dataloader):
    print(b)
    if i >= 2:
        break
