import pytest

from fastNLP.core.dataloaders.paddle_dataloader.fdl import PaddleDataLoader
from fastNLP.core.dataset import DataSet
from fastNLP.core.log import logger
from paddle.io import Dataset, DataLoader
import numpy as np
import paddle


class RandomDataset(Dataset):

    def __getitem__(self, idx):
        image = np.random.random((10, 5)).astype('float32')
        return {'image': image, 'label': [[0, 1], [1, 2, 3, 4]]}

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
            print(batch)
            # print(fdl.get_batch_indices())

    def test_fdl_batch_indices(self):
        ds = DataSet({'x': [[1, 2], [2, 3, 4], [1]] * 10, 'y': [0, 1, 1] * 10})
        fdl = PaddleDataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
        for batch in fdl:
            assert len(fdl.get_batch_indices()) == 4
            print(batch)
            print(fdl.get_batch_indices())

    def test_set_inputs_and_set_pad_val(self):
        logger.setLevel("DEBUG")
        ds = RandomDataset()
        fdl = PaddleDataLoader(ds, batch_size=2, drop_last=True)
        fdl.set_pad('label', -1)
        for batch in fdl:
            print(batch['image'])
            assert batch['image'].shape == [2, 10, 5]
            print(batch)
        fdl1 = PaddleDataLoader(ds, batch_size=4, drop_last=True)
        fdl1.set_ignore('image')
        for batch in fdl1:
            assert batch['image'].shape == [4, 10, 5]
            print(batch)

    def test_v2(self):
        from fastNLP.core.collators import Collator
        logger.setLevel("DEBUG")
        data = [paddle.Tensor(np.random.random((10, 5)).astype('float32')), paddle.Tensor(np.random.random((10, 5)).astype('float32'))]
        col = Collator(backend="jittor")
        res = col(data)
        print(res)