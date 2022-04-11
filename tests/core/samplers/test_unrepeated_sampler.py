from itertools import chain

import pytest

from fastNLP.core.samplers import UnrepeatedSampler, UnrepeatedSortedSampler


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
        sampler = UnrepeatedSampler(data, shuffle)
        indexes = set(sampler)
        assert indexes==set(range(num_of_data))

    @pytest.mark.parametrize('num_replica', [2, 3])
    @pytest.mark.parametrize('num_of_data', [2, 3, 4, 100])
    @pytest.mark.parametrize('shuffle', [False, True])
    def test_multi(self, num_replica, num_of_data, shuffle):
        data = DatasetWithVaryLength(num_of_data=num_of_data)
        samplers = []
        for i in range(num_replica):
            sampler = UnrepeatedSampler(dataset=data, shuffle=shuffle)
            sampler.set_distributed(num_replica, rank=i)
            samplers.append(sampler)

        indexes = set(chain(*samplers))
        assert indexes==set(range(num_of_data))


class TestUnrepeatedSortedSampler:
    @pytest.mark.parametrize('shuffle', [True, False])
    def test_single(self, shuffle):
        num_of_data = 100
        data = DatasetWithVaryLength(num_of_data)
        sampler = UnrepeatedSortedSampler(data, length=data.data)
        indexes = list(sampler)
        assert indexes==list(range(num_of_data-1, -1, -1))

    @pytest.mark.parametrize('num_replica', [2, 3])
    @pytest.mark.parametrize('num_of_data', [2, 3, 4, 100])
    @pytest.mark.parametrize('shuffle', [False, True])
    def test_multi(self, num_replica, num_of_data, shuffle):
        data = DatasetWithVaryLength(num_of_data=num_of_data)
        samplers = []
        for i in range(num_replica):
            sampler = UnrepeatedSortedSampler(dataset=data, length=data.data)
            sampler.set_distributed(num_replica, rank=i)
            samplers.append(sampler)

        indexes = set(chain(*samplers))
        assert indexes==set(range(num_of_data))
