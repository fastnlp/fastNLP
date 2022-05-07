import pytest
import numpy as np

from fastNLP.core.dataloaders.paddle_dataloader.fdl import PaddleDataLoader
from fastNLP.core.dataset import DataSet
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