from itertools import chain

import pytest

from fastNLP.core.samplers import UnrepeatedRandomSampler, UnrepeatedSortedSampler, UnrepeatedSequentialSampler


class DatasetWithVaryLength:
    def __init__(self, num_of_data=100):
        self.data = list(range(num_of_data))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class TestUnrepeatedSampler:
    @pytest.mark.parametrize('shuffle', [True, False])
    def test_single(self, shuffle):
        num_of_data = 100
        data = DatasetWithVaryLength(num_of_data)
        sampler = UnrepeatedRandomSampler(data, shuffle)
        indexes = set(sampler)
        assert indexes==set(range(num_of_data))

    @pytest.mark.parametrize('num_replica', [2, 3])
    @pytest.mark.parametrize('num_of_data', [2, 3, 4, 100])
    @pytest.mark.parametrize('shuffle', [False, True])
    def test_multi(self, num_replica, num_of_data, shuffle):
        data = DatasetWithVaryLength(num_of_data=num_of_data)
        samplers = []
        for i in range(num_replica):
            sampler = UnrepeatedRandomSampler(dataset=data, shuffle=shuffle)
            sampler.set_distributed(num_replica, rank=i)
            samplers.append(sampler)

        indexes = list(chain(*samplers))
        assert len(indexes) == num_of_data
        indexes = set(indexes)
        assert indexes==set(range(num_of_data))


class TestUnrepeatedSortedSampler:
    def test_single(self):
        num_of_data = 100
        data = DatasetWithVaryLength(num_of_data)
        sampler = UnrepeatedSortedSampler(data, length=data.data)
        indexes = list(sampler)
        assert indexes==list(range(num_of_data-1, -1, -1))

    @pytest.mark.parametrize('num_replica', [2, 3])
    @pytest.mark.parametrize('num_of_data', [2, 3, 4, 100])
    def test_multi(self, num_replica, num_of_data):
        data = DatasetWithVaryLength(num_of_data=num_of_data)
        samplers = []
        for i in range(num_replica):
            sampler = UnrepeatedSortedSampler(dataset=data, length=data.data)
            sampler.set_distributed(num_replica, rank=i)
            samplers.append(sampler)

        # 保证顺序是没乱的
        for sampler in samplers:
            prev_index = float('inf')
            for index in sampler:
                assert index <= prev_index
                prev_index = index

        indexes = list(chain(*samplers))
        assert len(indexes) == num_of_data  # 不同卡之间没有交叉
        indexes = set(indexes)
        assert indexes==set(range(num_of_data))


class TestUnrepeatedSequentialSampler:
    def test_single(self):
        num_of_data = 100
        data = DatasetWithVaryLength(num_of_data)
        sampler = UnrepeatedSequentialSampler(data, length=data.data)
        indexes = list(sampler)
        assert indexes==list(range(num_of_data))

    @pytest.mark.parametrize('num_replica', [2, 3])
    @pytest.mark.parametrize('num_of_data', [2, 3, 4, 100])
    def test_multi(self, num_replica, num_of_data):
        data = DatasetWithVaryLength(num_of_data=num_of_data)
        samplers = []
        for i in range(num_replica):
            sampler = UnrepeatedSequentialSampler(dataset=data, length=data.data)
            sampler.set_distributed(num_replica, rank=i)
            samplers.append(sampler)

        # 保证顺序是没乱的
        for sampler in samplers:
            prev_index = float('-inf')
            for index in sampler:
                assert index>=prev_index
                prev_index = index

        indexes = list(chain(*samplers))
        assert len(indexes) == num_of_data
        indexes = set(indexes)
        assert indexes == set(range(num_of_data))
