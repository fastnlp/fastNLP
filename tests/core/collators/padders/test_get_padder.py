import pytest
import numpy as np

from fastNLP.core.collators.padders.get_padder import get_padder, InconsistencyError, DtypeError, \
    _get_element_shape_dtype
from fastNLP.envs.imports import _NEED_IMPORT_TORCH, _NEED_IMPORT_PADDLE, _NEED_IMPORT_JITTOR


def test_get_element_shape_dtype():
    catalog = _get_element_shape_dtype([[1], [2, 3], [3], 2])
    catalog = _get_element_shape_dtype([['1'], [2, 3]])
    catalog = _get_element_shape_dtype([['1'], [2, 3]])
    catalog = _get_element_shape_dtype([['1'], ['2', '3']])
    catalog = _get_element_shape_dtype([np.zeros(3), np.zeros((2, 1))])


# @pytest.mark.parametrize('backend', ['raw', None, 'numpy', 'torch', 'jittor', 'paddle'])
@pytest.mark.parametrize('backend', ['raw', None, 'numpy', 'torch', 'paddle'])
@pytest.mark.torch
@pytest.mark.paddle
def test_get_padder_run(backend):
    if not _NEED_IMPORT_TORCH and backend == 'torch':
        pytest.skip("No torch")
    if not _NEED_IMPORT_PADDLE and backend == 'paddle':
        pytest.skip("No paddle")
    if not _NEED_IMPORT_JITTOR and backend == 'jittor':
        pytest.skip("No jittor")
    batch_field = [1, 2, 3]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')

    if backend is not None:
        # 不能 pad
        batch_field = [[1], [2, 3], [3], 2]
        with pytest.raises(InconsistencyError):
            padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
        padder = get_padder(batch_field, pad_val=None, backend=backend, dtype=int, field_name='test')

        # 不能 pad
        batch_field = [['2'], ['2'], ['2', '2']]
        with pytest.raises(DtypeError) as exec_info:
            padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
        padder = get_padder(batch_field, pad_val=None, backend=backend, dtype=int, field_name='test')

        batch_field = [np.zeros(3), np.zeros((3, 1))]
        with pytest.raises(InconsistencyError) as exec_info:
            padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
        padder = get_padder(batch_field, pad_val=None, backend=backend, dtype=int, field_name='test')  # no pad

        batch_field = [np.zeros((3, 1)), np.zeros((4, 1))]
        padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')


def test_raw_padder():
    backend = 'raw'
    batch_field = [1, 2, 3]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert pad_batch == batch_field

    batch_field = [[1], [2, 2], [3, 3, 3]]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert np.shape(pad_batch) == (3, 3)

    batch_field = [[[1]], [[2, 2], [2]], [[3], [3], [3]]]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert np.shape(pad_batch) == (3, 3, 2)

    batch_field = [np.ones((3,3)), np.ones((2,3)), np.ones((1,0))]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert isinstance(pad_batch, list)
    assert np.shape(pad_batch) == (3, 3, 3)
    assert (pad_batch == np.zeros(np.shape(pad_batch))).sum()==12


def test_numpy_padder():
    backend = 'numpy'
    target_type = np.ndarray
    batch_field = [1, 2, 3]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert isinstance(pad_batch, target_type)
    assert (pad_batch == np.array(batch_field)).sum()==len(batch_field)

    batch_field = [[1], [2, 2], [3, 3, 3]]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert isinstance(pad_batch, target_type)
    assert np.shape(pad_batch) == (3, 3)
    assert (pad_batch == np.zeros(np.shape(pad_batch))).sum()==3

    batch_field = [np.ones((3,3)), np.ones((2,3)), np.ones((1,3))]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert isinstance(pad_batch, target_type)
    assert np.shape(pad_batch) == (3, 3, 3)
    assert (pad_batch == np.zeros(np.shape(pad_batch))).sum()==9

    batch_field = [np.ones((3,3)), np.ones((2,3)), np.ones((1,0))]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert isinstance(pad_batch, target_type)
    assert np.shape(pad_batch) == (3, 3, 3)
    assert (pad_batch == np.zeros(np.shape(pad_batch))).sum()==12

    batch_field = [np.ones((3,3)), np.ones((2,3)), np.ones((1,))]
    with pytest.raises(InconsistencyError):
        padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')


@pytest.mark.torch
def test_torch_padder():
    if not _NEED_IMPORT_TORCH:
        pytest.skip("No torch.")
    import torch
    backend = 'torch'
    target_type = torch.Tensor
    batch_field = [1, 2, 3]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert isinstance(pad_batch, target_type)
    assert (pad_batch == torch.LongTensor(batch_field)).sum()==len(batch_field)

    batch_field = [[1], [2, 2], [3, 3, 3]]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert isinstance(pad_batch, target_type)
    assert pad_batch.shape == (3, 3)
    assert (pad_batch == torch.zeros(pad_batch.shape)).sum()==3

    batch_field = [torch.ones((3,3)), torch.ones((2,3)), torch.ones((1,3))]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert isinstance(pad_batch, target_type)
    assert pad_batch.shape == (3, 3, 3)
    assert (pad_batch == torch.zeros(pad_batch.shape)).sum()==9

    batch_field = [torch.ones((3,3)), torch.ones((2,3)), torch.ones((1,0))]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert isinstance(pad_batch, target_type)
    assert pad_batch.shape == (3, 3, 3)
    assert (pad_batch == torch.zeros(pad_batch.shape)).sum()==12

    batch_field = [torch.ones((3,3)), torch.ones((2,3)), torch.ones((1,))]
    with pytest.raises(InconsistencyError):
        padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')

    # 可以是 numpy.ndarray
    batch_field = [np.ones((3,3)), np.ones((2,3)), np.ones((1,0))]
    padder = get_padder(batch_field, pad_val=0, backend=backend, dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert isinstance(pad_batch, target_type)
    assert pad_batch.shape == (3, 3, 3)
    assert (pad_batch == torch.zeros(pad_batch.shape)).sum()==12

    # 测试 to numpy
    batch_field = [torch.ones((3,3)), torch.ones((2,3)), torch.ones((1,0))]
    padder = get_padder(batch_field, pad_val=0, backend='numpy', dtype=int, field_name='test')
    pad_batch = padder(batch_field)
    assert isinstance(pad_batch, np.ndarray)
    assert np.shape(pad_batch) == (3, 3, 3)
    assert (pad_batch == np.zeros(np.shape(pad_batch))).sum()==12
