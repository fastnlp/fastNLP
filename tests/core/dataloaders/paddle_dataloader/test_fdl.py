import pytest
import numpy as np

from fastNLP.core.dataloaders.paddle_dataloader.fdl import PaddleDataLoader, prepare_paddle_dataloader
from fastNLP.core.dataset import DataSet
from fastNLP.io.data_bundle import DataBundle
from fastNLP.core.log import logger
from fastNLP.core.collators import Collator

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE

if _NEED_IMPORT_PADDLE:
    from paddle.io import Dataset, DataLoader
    import paddle
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Dataset


class RandomDataset(Dataset):

    def __getitem__(self, idx):
        image = np.random.random((10, 5)).astype('float32')
        return {'image': paddle.to_tensor(image), 'label': [[0, 1], [1, 2, 3, 4]]}

    def __len__(self):
        return 10


@pytest.mark.paddle
class TestPaddle:

    def test_init(self):
        # ds = DataSet({'x': [[1, 2], [2, 3, 4], [1]] * 10, 'y': [0, 1, 1] * 10})
        ds = RandomDataset()
        fdl = PaddleDataLoader(ds, batch_size=2)
        # fdl = DataLoader(ds, batch_size=2, shuffle=True)
        for batch in fdl:
            assert batch['image'].shape == [2, 10, 5]
            assert batch['label'].shape == [2, 2, 4]
            # print(fdl.get_batch_indices())

    def test_fdl_fastnlp_dataset(self):
        ds = DataSet({'x': [[1, 2], [2, 3, 4], [1]] * 10, 'y': [0, 1, 1] * 10})
        fdl = PaddleDataLoader(ds, batch_size=3, shuffle=False, drop_last=True)
        fdl.set_ignore('y')
        fdl.set_pad('x', -1)
        for batch in fdl:
            assert len(fdl.get_batch_indices()) == 3
            assert 'y' not in batch
            assert batch['x'].shape == [3, 3]

        with pytest.raises(ValueError):
            PaddleDataLoader(ds, batch_size=3, collate_fn=None)

    def test_set_inputs_and_set_pad_val(self):
        logger.setLevel("DEBUG")
        ds = RandomDataset()
        fdl = PaddleDataLoader(ds, batch_size=2, drop_last=True)
        fdl.set_pad('label', -1)
        for batch in fdl:
            assert batch['image'].shape == [2, 10, 5]
        fdl1 = PaddleDataLoader(ds, batch_size=4, drop_last=True)
        fdl1.set_ignore('label')
        for batch in fdl1:
            assert batch['image'].shape == [4, 10, 5]

    def test_get_backend(self):
        ds = RandomDataset()
        collate_fn = Collator(backend='auto')
        paddle_dl = DataLoader(ds, collate_fn=collate_fn)
        for batch in paddle_dl:
            print(batch)

    def test_v4(self):
        from paddle.io import DataLoader
        from fastNLP import Collator
        from paddle.io import Dataset
        import paddle

        class PaddleArgMaxDataset(Dataset):
            def __init__(self, num_samples, num_features):
                self.x = paddle.randn((num_samples, num_features))
                self.y = self.x.argmax(axis=-1)

            def __len__(self):
                return len(self.x)

            def __getitem__(self, item):
                return {"x": self.x[item], "y": self.y[item]}

        ds = PaddleArgMaxDataset(100, 2)
        dl = DataLoader(ds, places=None, collate_fn=Collator(), batch_size=4)
        for batch in dl:
            print(batch)

    def test_prepare_paddle_dataloader(self):
        # 测试 fastNLP 的 dataset
        ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        dl = prepare_paddle_dataloader(ds, batch_size=8, shuffle=True, num_workers=2)
        assert isinstance(dl, PaddleDataLoader)

        ds1 = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        dbl = DataBundle(datasets={'train': ds, 'val': ds1})
        dl_bundle = prepare_paddle_dataloader(dbl)
        assert isinstance(dl_bundle['train'], PaddleDataLoader)
        assert isinstance(dl_bundle['val'], PaddleDataLoader)

        ds_dict = {'train_1': ds, 'val': ds1}
        dl_dict = prepare_paddle_dataloader(ds_dict)
        assert isinstance(dl_dict['train_1'], PaddleDataLoader)
        assert isinstance(dl_dict['val'], PaddleDataLoader)

        ds2 = RandomDataset()
        dl1 = prepare_paddle_dataloader(ds2, batch_size=8, shuffle=True, num_workers=2)
        assert isinstance(dl1, PaddleDataLoader)

        ds3 = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
        dbl1 = DataBundle(datasets={'train': ds2, 'val': ds3})
        dl_bundle1 = prepare_paddle_dataloader(dbl1)
        assert isinstance(dl_bundle1['train'], PaddleDataLoader)
        assert isinstance(dl_bundle1['val'], PaddleDataLoader)

        ds_dict1 = {'train_1': ds2, 'val': ds3}
        dl_dict1 = prepare_paddle_dataloader(ds_dict1)
        assert isinstance(dl_dict1['train_1'], PaddleDataLoader)
        assert isinstance(dl_dict1['val'], PaddleDataLoader)
