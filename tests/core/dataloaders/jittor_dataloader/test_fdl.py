import pytest
import numpy as np
from datasets import Dataset as HfDataset
from datasets import load_dataset

from fastNLP.core.dataloaders.jittor_dataloader import JittorDataLoader
from fastNLP.core.dataset import DataSet as Fdataset
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
if _NEED_IMPORT_JITTOR:
    from jittor.dataset import Dataset
    import jittor
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Dataset


class MyDataset(Dataset):

    def __init__(self, data_len=1000):
        super(MyDataset, self).__init__()
        self.data = [jittor.ones((3, 4)) for _ in range(data_len)]
        self.set_attrs(total_len=data_len)
        self.dataset_len = data_len

    def __getitem__(self, item):
        return self.data[item]
        # return {'x': [[1, 0], [2, 0, 1]]}
        # return np.random.randn(3, 10)

    # def __len__(self):
    #     return self.dataset_len

@pytest.mark.jittor
class TestJittor:

    def test_v1(self):
        """
        测试jittor类型的dataset使用fdl

        :return:
        """
        dataset = MyDataset()
        jtl = JittorDataLoader(dataset, keep_numpy_array=True, batch_size=4)
        # jtl.set_pad_val('x', 'y')
        # jtl.set_input('x')
        for batch in jtl:
            print(batch)
            print(jtl.get_batch_indices())

    def test_v2(self):
        """
        测试fastnlp的dataset

        :return:
        """
        dataset = Fdataset({'x': [[1, 2], [0], [2, 3, 4, 5]] * 100, 'y': [0, 1, 2] * 100})
        jtl = JittorDataLoader(dataset, batch_size=16, drop_last=True)
        jtl.set_pad("x", -1)
        jtl.set_ignore("y")
        # jtl.set_pad_val('x', val=-1)
        # jtl.set_input('x', 'y')
        for batch in jtl:
            assert batch['x'].size() == (16, 4)

    def test_v3(self):
        dataset = HfDataset.from_dict({'x': [[1, 2], [0], [2, 3, 4, 5]] * 100, 'y': [0, 1, 2] * 100})
        jtl = JittorDataLoader(dataset, batch_size=4, drop_last=True)
        # jtl.set_input('x', 'y')
        for batch in jtl:
            print(batch)

    def test_v4(self):
        dataset = MyDataset()
        dl = JittorDataLoader(dataset, batch_size=4, num_workers=2)
        print(len(dl))
        for idx, batch in enumerate(dl):
            print(batch.shape, idx)
        for idx, batch in enumerate(dl):
            print(batch.shape, idx)

    def test_v5(self):
        dataset = MyDataset()
        dataset.set_attrs(batch_size=4, num_workers=2)
        for idx, batch in enumerate(dataset):
            print(idx, batch.shape)
        for idx, batch in enumerate(dataset):
            print(idx, batch.shape)