import pytest
import numpy as np
from fastNLP.envs import _module_available

if _module_available('datasets'):
    from datasets import Dataset as HfDataset

from fastNLP.core.dataloaders.jittor_dataloader import JittorDataLoader, prepare_jittor_dataloader
from fastNLP.core.dataset import DataSet as Fdataset
from fastNLP.core.collators import Collator
from fastNLP.io.data_bundle import DataBundle
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


@pytest.mark.jittor
class TestJittor:

    def test_jittor_dataset(self):
        """
        测试jittor类型的dataset使用fdl

        :return:
        """
        dataset = MyDataset()
        jtl = JittorDataLoader(dataset, keep_numpy_array=False, batch_size=4)
        for batch in jtl:
            assert batch.size() == [4, 3, 4]
        jtl1 = JittorDataLoader(dataset, keep_numpy_array=False, batch_size=4, num_workers=2)
        for batch in jtl1:
            assert batch.size() == [4, 3, 4]

    def test_fastnlp_Dataset(self):
        """
        测试fastnlp的dataset

        :return:
        """
        dataset = Fdataset({'x': [[1, 2], [0], [2, 3, 4, 5]] * 100, 'y': [0, 1, 2] * 100})
        # jtl = JittorDataLoader(dataset, batch_size=16, drop_last=True)
        # jtl.set_pad("x", -1)
        # jtl.set_ignore("y")
        # for batch in jtl:
        #     assert batch['x'].size() == (16, 4)
        jtl1 = JittorDataLoader(dataset, batch_size=16, drop_last=True, num_workers=2)
        for batch in jtl1:
            print(batch)

    def test_huggingface_datasets(self):
        dataset = HfDataset.from_dict({'x': [[1, 2], [0], [2, 3, 4, 5]] * 100, 'y': [0, 1, 2] * 100})
        jtl = JittorDataLoader(dataset, batch_size=4, drop_last=True, shuffle=False)
        for batch in jtl:
            assert batch['x'].size() == [4, 4]
            assert len(batch['y']) == 4

    def test_num_workers(self):
        dataset = MyDataset()
        dl = JittorDataLoader(dataset, batch_size=4, num_workers=2)
        for idx, batch in enumerate(dl):
            assert batch.shape == [4, 3, 4]
        for idx, batch in enumerate(dl):
            assert batch.shape == [4, 3, 4]

    def test_v5(self):
        dataset = MyDataset()
        dataset.set_attrs(batch_size=4, num_workers=2)
        for idx, batch in enumerate(dataset):
            print(idx, batch.shape)
        for idx, batch in enumerate(dataset):
            print(idx, batch.shape)

    def test_jittor_get_backend(self):
        collate_bacth = Collator(backend='auto')
        dl = MyDataset()
        dl = dl.set_attrs(collate_batch=collate_bacth, batch_size=256)
        for batch in dl:
            print(batch)

    def test_prepare_jittor_dataloader(self):
        # 测试 fastnlp 的 dataset
        ds = Fdataset({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        dl = prepare_jittor_dataloader(ds, batch_size=8, shuffle=True, num_workers=2)
        assert isinstance(dl, JittorDataLoader)

        ds1 = Fdataset({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        dbl = DataBundle(datasets={'train': ds, 'val': ds1})
        dl_bundle = prepare_jittor_dataloader(dbl)
        assert isinstance(dl_bundle['train'], JittorDataLoader)
        assert isinstance(dl_bundle['val'], JittorDataLoader)

        ds_dict = {'train_1': ds, 'val': ds1}
        dl_dict = prepare_jittor_dataloader(ds_dict)
        assert isinstance(dl_dict['train_1'], JittorDataLoader)
        assert isinstance(dl_dict['val'], JittorDataLoader)

        # 测试 jittor 的 dataset
        ds1 = MyDataset()
        dl1 = prepare_jittor_dataloader(ds1, batch_size=8, shuffle=True, num_workers=2)
        assert isinstance(dl1, JittorDataLoader)

        ds2 = MyDataset()
        dbl1 = DataBundle(datasets={'train': ds1, 'val': ds2})
        dl_bundle1 = prepare_jittor_dataloader(dbl1)
        assert isinstance(dl_bundle1['train'], JittorDataLoader)
        assert isinstance(dl_bundle1['val'], JittorDataLoader)

        ds_dict1 = {'train_1': ds1, 'val': ds2}
        dl_dict1 = prepare_jittor_dataloader(ds_dict1)
        assert isinstance(dl_dict1['train_1'], JittorDataLoader)
        assert isinstance(dl_dict1['val'], JittorDataLoader)
