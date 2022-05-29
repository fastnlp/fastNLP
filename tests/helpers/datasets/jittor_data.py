from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    import jittor as jt
    from jittor.dataset import Dataset
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Dataset

class JittorNormalDataset(Dataset):
    def __init__(self, num_of_data=100, **kwargs):
        super(JittorNormalDataset, self).__init__(**kwargs)
        self._data = list(range(num_of_data))
        self.set_attrs(total_len=num_of_data)

    def __getitem__(self, item):
        return self._data[item]

class JittorNormalXYDataset(Dataset):
    """
    可以被输入到分类模型中的普通数据集
    """
    def __init__(self, num_of_data=1000, **kwargs):
        super(JittorNormalXYDataset, self).__init__(**kwargs)
        self.num_of_data = num_of_data
        self._data = list(range(num_of_data))
        self.set_attrs(total_len=num_of_data)

    def __getitem__(self, item):
        return {
            "x": jt.Var([self._data[item]]),
            "y": jt.Var([self._data[item]])
        }

class JittorArgMaxDataset(Dataset):
    def __init__(self, num_samples, num_features, **kwargs):
        super(JittorArgMaxDataset, self).__init__(**kwargs)
        self.x = jt.randn(num_samples, num_features)
        self.y = self.x.argmax(dim=-1)
        self.set_attrs(total_len=num_samples)

    def __getitem__(self, item):
        return {"x": self.x[item], "y": self.y[item]}

if __name__ == "__main__":
    dataset = JittorNormalDataset()
    print(len(dataset))