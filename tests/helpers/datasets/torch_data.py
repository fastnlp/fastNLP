from functools import reduce
from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch
    from torch.utils.data import Dataset
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Dataset


class TorchNormalDataset(Dataset):
    def __init__(self, num_of_data=1000):
        self.num_of_data = num_of_data
        self._data = list(range(num_of_data))

    def __len__(self):
        return self.num_of_data

    def __getitem__(self, item):
        return self._data[item]


# 该类专门用于为 tests.helpers.models.torch_model.py/ TorchNormalModel_Classification_1 创建数据；
class TorchNormalDataset_Classification(Dataset):
    def __init__(self, num_labels, feature_dimension=2, each_label_data=1000, seed=0):
        self.num_labels = num_labels
        self.feature_dimension = feature_dimension
        self.each_label_data = each_label_data
        self.seed = seed

        torch.manual_seed(seed)
        self.x_center = torch.randint(low=-100, high=100, size=[num_labels, feature_dimension])
        random_shuffle = torch.randn([num_labels, each_label_data, feature_dimension]) / 10
        self.x = self.x_center.unsqueeze(1).expand(num_labels, each_label_data, feature_dimension) + random_shuffle
        self.x = self.x.view(num_labels * each_label_data, feature_dimension)
        self.y = reduce(lambda x, y: x+y, [[i] * each_label_data for i in range(num_labels)])

    def __len__(self):
        return self.num_labels * self.each_label_data

    def __getitem__(self, item):
        return {"x": self.x[item], "y": self.y[item]}


class TorchArgMaxDataset(Dataset):
    def __init__(self, feature_dimension=10, data_num=1000, seed=0):
        self.num_labels = feature_dimension
        self.feature_dimension = feature_dimension
        self.data_num = data_num
        self.seed = seed

        g = torch.Generator()
        g.manual_seed(1000)
        self.x = torch.randint(low=-100, high=100, size=[data_num, feature_dimension], generator=g).float()
        self.y = torch.max(self.x, dim=-1)[1]

    def __len__(self):
        return self.data_num

    def __getitem__(self, item):
        return {"x": self.x[item], "y": self.y[item]}


if __name__ == "__main__":
    a = TorchNormalDataset_Classification(2, each_label_data=4)

    print(a.x)
    print(a.y)
    print(a[0])



