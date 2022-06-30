import pytest

from fastNLP.core.dataloaders.oneflow_dataloader import OneflowDataLoader, prepare_oneflow_dataloader
from fastNLP.core.dataset import DataSet
from fastNLP.io.data_bundle import DataBundle
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW
from tests.helpers.utils import Capturing, recover_logger
from fastNLP import logger
import numpy as np

if _NEED_IMPORT_ONEFLOW:
    import oneflow


@pytest.mark.oneflow
class TestFdl:

    def test_init_v1(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        fdl = OneflowDataLoader(ds, batch_size=3, shuffle=True, drop_last=True)
        # for batch in fdl:
        #     print(batch)
        fdl1 = OneflowDataLoader(ds, batch_size=3, shuffle=True, drop_last=True)
        # for batch in fdl1:
        #     print(batch)

    def test_set_padding(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        fdl = OneflowDataLoader(ds, batch_size=3)
        fdl.set_pad("x", -1)
        for batch in fdl:
            assert batch['x'].shape == oneflow.Size([3, 4])

    def test_get_batch_indices(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        fdl = OneflowDataLoader(ds, batch_size=3, shuffle=True)
        for batch in fdl:
            assert len(fdl.get_batch_indices()) == 3

    def test_other_dataset(self):
        import numpy as np
        class _DataSet:

            def __init__(self):
                pass

            def __getitem__(self, item):
                return np.random.randn(5), [[1, 2], [2, 3, 4]]

            def __len__(self):
                return 10

            def __getattribute__(self, item):
                return object.__getattribute__(self, item)

        dataset = _DataSet()
        dl = OneflowDataLoader(dataset, batch_size=2, shuffle=True)
        # dl.set_inputs('data', 'labels')
        # dl.set_pad_val('labels', val=None)
        for batch in dl:
            assert batch[0].shape == oneflow.Size([2, 5])
            assert batch[1].shape == oneflow.Size([2, 2, 3])

    def test_default_collate_fn(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        with pytest.raises(ValueError):
            fdl = OneflowDataLoader(ds, batch_size=3, collate_fn=None)
        import numpy as np
        class _DataSet:

            def __init__(self):
                pass

            def __getitem__(self, item):
                return np.random.randn(5), [[1, 2], [2, 3, 4]]

            def __len__(self):
                return 10

        fdl = OneflowDataLoader(_DataSet(), batch_size=3, collate_fn=None, drop_last=True)
        for batch in fdl:
            assert batch[0].shape == oneflow.Size([3, 5])

    def test_my_collate_fn(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        def collate_fn(batch):
            res = {'x': [], 'y': []}
            for ins in batch:
                res['x'].append(ins['x'])
                res['y'].append(ins['y'])
            return res
        fdl = OneflowDataLoader(ds, collate_fn=collate_fn, batch_size=3, drop_last=True)
        for batch in fdl:
            assert batch['x'] == [[1, 2], [2, 3, 4], [4, 5, 6, 7]]
            assert batch['y'] == [1, 0, 1]

    def test_prepare_oneflow_dataloader(self):
        # 测试 fastNLP 的 dataset
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        dl = prepare_oneflow_dataloader(ds, batch_size=8, shuffle=True, num_workers=2)
        assert isinstance(dl, OneflowDataLoader)

        ds1 = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        dbl = DataBundle(datasets={'train': ds, 'val': ds1})
        dl_bundle = prepare_oneflow_dataloader(dbl)
        assert isinstance(dl_bundle['train'], OneflowDataLoader)
        assert isinstance(dl_bundle['val'], OneflowDataLoader)

        ds_dict = {'train_1': ds, 'val': ds1}
        dl_dict = prepare_oneflow_dataloader(ds_dict)
        assert isinstance(dl_dict['train_1'], OneflowDataLoader)
        assert isinstance(dl_dict['val'], OneflowDataLoader)

        # 测试其他 dataset
        class _DataSet:

            def __init__(self):
                pass

            def __getitem__(self, item):
                return np.random.randn(5), [[1, 2], [2, 3, 4]]

            def __len__(self):
                return 10

            def __getattribute__(self, item):
                return object.__getattribute__(self, item)

        ds2 = _DataSet()
        dl1 = prepare_oneflow_dataloader(ds2, batch_size=8, shuffle=True, num_workers=2)
        assert isinstance(dl1, OneflowDataLoader)

        ds3 = _DataSet()
        dbl1 = DataBundle(datasets={'train': ds2, 'val': ds3})
        dl_bundle1 = prepare_oneflow_dataloader(dbl1)
        assert isinstance(dl_bundle1['train'], OneflowDataLoader)
        assert isinstance(dl_bundle1['val'], OneflowDataLoader)

        ds_dict1 = {'train_1': ds2, 'val': ds3}
        dl_dict1 = prepare_oneflow_dataloader(ds_dict1)
        assert isinstance(dl_dict1['train_1'], OneflowDataLoader)
        assert isinstance(dl_dict1['val'], OneflowDataLoader)

        ds = [[1, [1]], [2, [2, 2]]]
        dl = prepare_oneflow_dataloader(ds, batch_size=2)
        for batch in dl:
            assert (batch[0] == oneflow.LongTensor([1, 2])).sum()==2
            assert (batch[1] == oneflow.LongTensor([[1, 0], [2, 2]])).sum()==4

        # sequence = [ds, ds1]
        # seq_ds = prepare_oneflow_dataloader(sequence)
        # assert isinstance(seq_ds[0], OneflowDataLoader)
        # assert isinstance(seq_ds[1], OneflowDataLoader)

    def test_get_backend(self):
        from fastNLP.core.collators import Collator
        from oneflow.utils.data import DataLoader, Dataset

        class MyDatset(DataSet):
            def __len__(self):
                return 1000

            def __getitem__(self, item):
                return [[1, 0], [1], [1, 2, 4]], [1, 0]

        collate_batch = Collator(backend='auto')
        dl = DataLoader(MyDatset(), collate_fn=collate_batch)
        for batch in dl:
            print(batch)
