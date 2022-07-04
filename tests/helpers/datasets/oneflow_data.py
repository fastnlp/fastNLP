from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW

if _NEED_IMPORT_ONEFLOW:
    import oneflow
    from oneflow.utils.data import Dataset
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Dataset


class OneflowNormalDataset(Dataset):
    def __init__(self, num_of_data=1000):
        self.num_of_data = num_of_data
        self._data = list(range(num_of_data))

    def __len__(self):
        return self.num_of_data

    def __getitem__(self, item):
        return self._data[item]

class OneflowNormalXYDataset(Dataset):
    """
    可以被输入到分类模型中的普通数据集
    """
    def __init__(self, num_of_data=1000):
        self.num_of_data = num_of_data
        self._data = list(range(num_of_data))

    def __len__(self):
        return self.num_of_data

    def __getitem__(self, item):
        return {
            "x": oneflow.tensor([self._data[item]], dtype=oneflow.float),
            "y": oneflow.tensor([self._data[item]], dtype=oneflow.float)
        }


class OneflowArgMaxDataset(Dataset):
    def __init__(self, data_num=1000, feature_dimension=10, seed=0):
        self.num_labels = feature_dimension
        self.feature_dimension = feature_dimension
        self.data_num = data_num
        self.seed = seed

        g = oneflow.Generator()
        g.manual_seed(1000)
        self.x = oneflow.randint(low=-100, high=100, size=[data_num, feature_dimension], generator=g).float()
        self.y = oneflow.max(self.x, dim=-1)[1]

    def __len__(self):
        return self.data_num

    def __getitem__(self, item):
        return {"x": self.x[item], "y": self.y[item]}
