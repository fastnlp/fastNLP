import pytest

from fastNLP import prepare_dataloader
from fastNLP import DataSet
from fastNLP.io import DataBundle


@pytest.mark.torch
def test_torch():
    import torch
    ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
    dl = prepare_dataloader(ds, batch_size=2, shuffle=True)
    for batch in dl:
        assert isinstance(batch['x'], torch.Tensor)


@pytest.mark.torch
def test_torch_data_bundle():
    import torch
    ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
    dl = DataBundle()
    dl.set_dataset(dataset=ds, name='train')
    dl.set_dataset(dataset=ds, name='test')
    dls = prepare_dataloader(dl, batch_size=2, shuffle=True)
    for dl in dls.values():
        for batch in dl:
            assert isinstance(batch['x'], torch.Tensor)
            assert batch['x'].size(0) == 2
