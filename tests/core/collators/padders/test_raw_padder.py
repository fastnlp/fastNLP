import numpy as np
import pytest

from fastNLP.core.collators.padders.raw_padder import RawNumberPadder, RawSequencePadder
from fastNLP.core.collators.padders.exceptions import DtypeError


class TestRawNumberPadder:
    def test_run(self):
        padder = RawNumberPadder(ele_dtype=int, dtype=int, pad_val=-1)
        a = [1, 2, 3]
        assert padder(a) == a


class TestRawSequencePadder:
    def test_run(self):
        padder = RawSequencePadder(ele_dtype=int, dtype=int, pad_val=-1)
        a = [[1, 2, 3], [3]]
        a = padder(a)
        shape = np.shape(a)
        assert shape == (2, 3)
        b = np.array([[1, 2, 3], [3, -1, -1]])
        assert (a == b).sum().item() == shape[0]*shape[1]

    def test_dtype_check(self):
        with pytest.raises(DtypeError):
            padder = RawSequencePadder(ele_dtype=np.zeros(3, dtype=np.int8).dtype, dtype=int, pad_val=-1)
        with pytest.raises(DtypeError):
            padder = RawSequencePadder(ele_dtype=str, dtype=int, pad_val=-1)