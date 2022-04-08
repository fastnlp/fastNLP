import unittest

from fastNLP.core.dataloaders.torch_dataloader import FDataLoader, prepare_dataloader
from fastNLP.core.dataset import DataSet
from fastNLP.io.data_bundle import DataBundle


class TestFdl(unittest.TestCase):

    def test_init_v1(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        fdl = FDataLoader(ds, batch_size=3, shuffle=True, drop_last=True)
        # for batch in fdl:
        #     print(batch)
        fdl1 = FDataLoader(ds, batch_size=3, shuffle=True, drop_last=True, as_numpy=True)
        # for batch in fdl1:
        #     print(batch)

    def test_set_padding(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        ds.set_pad_val("x", val=-1)
        fdl = FDataLoader(ds, batch_size=3)
        fdl.set_input("x", "y")
        for batch in fdl:
            print(batch)
        fdl.set_pad_val("x", val=-2)
        for batch in fdl:
            print(batch)

    def test_add_collator(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})

        def collate_fn(ins_list):
            _dict = {"Y": []}
            for ins in ins_list:
                _dict["Y"].append(ins['y'])
            return _dict

        fdl = FDataLoader(ds, batch_size=3, as_numpy=True)
        fdl.set_input("x", "y")
        fdl.add_collator(collate_fn)
        for batch in fdl:
            print(batch)

    def test_get_batch_indices(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        fdl = FDataLoader(ds, batch_size=3, shuffle=True)
        fdl.set_input("y", "x")
        for batch in fdl:
            print(fdl.get_batch_indices())

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
        dl = FDataLoader(dataset, batch_size=2, shuffle=True)
        # dl.set_inputs('data', 'labels')
        # dl.set_pad_val('labels', val=None)
        for batch in dl:
            print(batch)
            print(dl.get_batch_indices())

    def test_prepare_dataloader(self):
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        dl = prepare_dataloader(ds, batch_size=8, shuffle=True, num_workers=2)
        assert isinstance(dl, FDataLoader)

        ds1 = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        dbl = DataBundle(datasets={'train': ds, 'val': ds1})
        dl_bundle = prepare_dataloader(dbl)
        assert isinstance(dl_bundle['train'], FDataLoader)
        assert isinstance(dl_bundle['val'], FDataLoader)

        ds_dict = {'train_1': ds, 'val': ds1}
        dl_dict = prepare_dataloader(ds_dict)
        assert isinstance(dl_dict['train_1'], FDataLoader)
        assert isinstance(dl_dict['val'], FDataLoader)

        sequence = [ds, ds1]
        seq_ds = prepare_dataloader(sequence)
        assert isinstance(seq_ds[0], FDataLoader)
        assert isinstance(seq_ds[1], FDataLoader)
