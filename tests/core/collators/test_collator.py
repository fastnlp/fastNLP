import pytest

from fastNLP.core.collators import AutoCollator
from fastNLP.core.collators.collator import _MultiCollator
from fastNLP.core.dataset import DataSet


class TestCollator:

    @pytest.mark.parametrize('as_numpy', [True, False])
    def test_auto_collator(self, as_numpy):
        """
        测试auto_collator的auto_pad功能

        :param as_numpy:
        :return:
        """
        dataset = DataSet({'x': [[1, 2], [0, 1, 2, 3], [3], [9, 0, 10, 1, 5]] * 100,
                           'y': [0, 1, 1, 0] * 100})
        collator = AutoCollator(as_numpy=as_numpy)
        collator.set_input('x', 'y')
        bucket_data = []
        data = []
        for i in range(len(dataset)):
            data.append(dataset[i])
            if len(data) == 40:
                bucket_data.append(data)
                data = []
        results = []
        for bucket in bucket_data:
            res = collator(bucket)
            assert res['x'].shape == (40, 5)
            assert res['y'].shape == (40,)
            results.append(res)

    def test_auto_collator_v1(self):
        """
        测试auto_collator的set_pad_val和set_pad_val功能

        :return:
        """
        dataset = DataSet({'x': [[1, 2], [0, 1, 2, 3], [3], [9, 0, 10, 1, 5]] * 100,
                           'y': [0, 1, 1, 0] * 100})
        collator = AutoCollator(as_numpy=False)
        collator.set_input('x')
        collator.set_pad_val('x', val=-1)
        collator.set_as_numpy(True)
        bucket_data = []
        data = []
        for i in range(len(dataset)):
            data.append(dataset[i])
            if len(data) == 40:
                bucket_data.append(data)
                data = []
        for bucket in bucket_data:
            res = collator(bucket)
            print(res)

    def test_multicollator(self):
        """
        测试multicollator功能

        :return:
        """
        dataset = DataSet({'x': [[1, 2], [0, 1, 2, 3], [3], [9, 0, 10, 1, 5]] * 100,
                           'y': [0, 1, 1, 0] * 100})
        collator = AutoCollator(as_numpy=False)
        multi_collator = _MultiCollator(collator)
        multi_collator.set_as_numpy(as_numpy=True)
        multi_collator.set_pad_val('x', val=-1)
        multi_collator.set_input('x')
        bucket_data = []
        data = []
        for i in range(len(dataset)):
            data.append(dataset[i])
            if len(data) == 40:
                bucket_data.append(data)
                data = []
        for bucket in bucket_data:
            res = multi_collator(bucket)
            print(res)
