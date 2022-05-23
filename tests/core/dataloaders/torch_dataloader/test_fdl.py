import pytest

from fastNLP.core.dataloaders.torch_dataloader import TorchDataLoader, prepare_torch_dataloader
from fastNLP.core.dataset import DataSet
from fastNLP.io.data_bundle import DataBundle
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.core import Trainer
from pkg_resources import parse_version
from tests.helpers.utils import Capturing, recover_logger
from fastNLP import logger

if _NEED_IMPORT_TORCH:
    import torch


@pytest.mark.torch
class TestFdl:

    def test_init_v1(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        fdl = TorchDataLoader(ds, batch_size=3, shuffle=True, drop_last=True)
        # for batch in fdl:
        #     print(batch)
        fdl1 = TorchDataLoader(ds, batch_size=3, shuffle=True, drop_last=True)
        # for batch in fdl1:
        #     print(batch)

    def test_set_padding(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        fdl = TorchDataLoader(ds, batch_size=3)
        fdl.set_pad("x", -1)
        for batch in fdl:
            assert batch['x'].shape == torch.Size([3, 4])

    def test_get_batch_indices(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        fdl = TorchDataLoader(ds, batch_size=3, shuffle=True)
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
        dl = TorchDataLoader(dataset, batch_size=2, shuffle=True)
        # dl.set_inputs('data', 'labels')
        # dl.set_pad_val('labels', val=None)
        for batch in dl:
            assert batch[0].shape == torch.Size([2, 5])
            assert batch[1].shape == torch.Size([2, 2, 3])

    def test_default_collate_fn(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        with pytest.raises(ValueError):
            fdl = TorchDataLoader(ds, batch_size=3, collate_fn=None)
        import numpy as np
        class _DataSet:

            def __init__(self):
                pass

            def __getitem__(self, item):
                return np.random.randn(5), [[1, 2], [2, 3, 4]]

            def __len__(self):
                return 10

        fdl = TorchDataLoader(_DataSet(), batch_size=3, collate_fn=None, drop_last=True)
        for batch in fdl:
            assert batch[0].shape == torch.Size([3, 5])

    def test_my_collate_fn(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        def collate_fn(batch):
            res = {'x': [], 'y': []}
            for ins in batch:
                res['x'].append(ins['x'])
                res['y'].append(ins['y'])
            return res
        fdl = TorchDataLoader(ds, collate_fn=collate_fn, batch_size=3, drop_last=True)
        for batch in fdl:
            assert batch['x'] == [[1, 2], [2, 3, 4], [4, 5, 6, 7]]
            assert batch['y'] == [1, 0, 1]

    def test_prepare_torch_dataloader(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        dl = prepare_torch_dataloader(ds, batch_size=8, shuffle=True, num_workers=2)
        assert isinstance(dl, TorchDataLoader)

        ds1 = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        dbl = DataBundle(datasets={'train': ds, 'val': ds1})
        dl_bundle = prepare_torch_dataloader(dbl)
        assert isinstance(dl_bundle['train'], TorchDataLoader)
        assert isinstance(dl_bundle['val'], TorchDataLoader)

        ds_dict = {'train_1': ds, 'val': ds1}
        dl_dict = prepare_torch_dataloader(ds_dict)
        assert isinstance(dl_dict['train_1'], TorchDataLoader)
        assert isinstance(dl_dict['val'], TorchDataLoader)

        sequence = [ds, ds1]
        seq_ds = prepare_torch_dataloader(sequence)
        assert isinstance(seq_ds[0], TorchDataLoader)
        assert isinstance(seq_ds[1], TorchDataLoader)

    def test_get_backend(self):
        from fastNLP.core.collators import Collator
        from torch.utils.data import DataLoader, Dataset

        class MyDatset(DataSet):
            def __len__(self):
                return 1000

            def __getitem__(self, item):
                return [[1, 0], [1], [1, 2, 4]], [1, 0]

        collate_batch = Collator(backend='auto')
        dl = DataLoader(MyDatset(), collate_fn=collate_batch)
        for batch in dl:
            print(batch)

    @recover_logger
    def test_version_16(self):
        if parse_version(torch.__version__) >= parse_version('1.7'):
            pytest.skip("Torch version larger than 1.7")
        logger.set_stdout()
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        with Capturing() as out:
            dl = TorchDataLoader(ds, batch_size=1, prefetch_factor=3, shuffle=False)
            for idx, batch in enumerate(dl):
                assert len(batch['x'])==1
                assert batch['x'][0].tolist() == ds[idx]['x']

        assert 'Parameter:prefetch_factor' in out[0]

    @recover_logger
    def test_version_111(self):
        if parse_version(torch.__version__) <= parse_version('1.7'):
            pytest.skip("Torch version smaller than 1.7")
        logger.set_stdout()
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        with Capturing() as out:
            dl = TorchDataLoader(ds, batch_size=1, num_workers=0, prefetch_factor=2, generator=torch.Generator(), shuffle=False)
            for idx, batch in enumerate(dl):
                assert len(batch['x'])==1
                assert batch['x'][0].tolist() == ds[idx]['x']

        assert 'Parameter:prefetch_factor' not in out[0]


