import pytest

from fastNLP import prepare_dataloader
from fastNLP import DataSet


@pytest.mark.torch
def test_torch():
    import torch
    ds = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})
    dl = prepare_dataloader(ds, batch_size=2, shuffle=True)
    for batch in dl:
        assert isinstance(batch['x'], torch.Tensor)