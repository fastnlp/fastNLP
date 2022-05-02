import pytest

from fastNLP.core.drivers.torch_driver.utils import (
    replace_batch_sampler,
    replace_sampler,
)
from fastNLP.core.samplers import ReproduceBatchSampler, RandomSampler
from torch.utils.data import DataLoader, BatchSampler

from tests.helpers.datasets.torch_data import TorchNormalDataset


@pytest.mark.torch
def test_replace_batch_sampler():
    dataset = TorchNormalDataset(10)
    dataloader = DataLoader(dataset, batch_size=32)
    batch_sampler = ReproduceBatchSampler(dataloader.batch_sampler, batch_size=16, drop_last=False)

    replaced_loader = replace_batch_sampler(dataloader, batch_sampler)

    assert not (replaced_loader is dataloader)
    assert isinstance(replaced_loader.batch_sampler, ReproduceBatchSampler)
    assert isinstance(replaced_loader.dataset, TorchNormalDataset)
    assert len(replaced_loader.dataset) == len(dataset)
    assert replaced_loader.batch_sampler.batch_size == 16


@pytest.mark.torch
def test_replace_sampler():
    dataset = TorchNormalDataset(10)
    dataloader = DataLoader(dataset, batch_size=32)
    sampler = RandomSampler(dataset)

    replaced_loader = replace_sampler(dataloader, sampler)

    assert not (replaced_loader is dataloader)
    assert isinstance(replaced_loader.batch_sampler, BatchSampler)
    assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)