import pytest

from fastNLP.core.drivers.jittor_driver.utils import replace_sampler
from fastNLP.core.samplers import ReproduceBatchSampler, RandomSampler
from fastNLP.core.dataloaders import JittorDataLoader
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
if _NEED_IMPORT_JITTOR:
    import jittor as jt

from tests.helpers.datasets.jittor_data import JittorNormalDataset

@pytest.mark.jittor
@pytest.mark.parametrize("dataset", [
    JittorNormalDataset(20, batch_size=10, shuffle=True),
    JittorNormalDataset(20, batch_size=5, drop_last=True),
    JittorNormalDataset(20)
])
def test_replace_sampler_dataset(dataset):
    dataset = JittorNormalDataset(20)
    sampler = RandomSampler(dataset)

    replaced_loader = replace_sampler(dataset, sampler)

    assert not (replaced_loader is dataset)
    assert isinstance(replaced_loader.sampler, RandomSampler)
    assert replaced_loader.batch_size == dataset.batch_size
    assert replaced_loader.drop_last == dataset.drop_last
    assert replaced_loader.shuffle == dataset.shuffle
    assert replaced_loader.total_len == dataset.total_len

@pytest.mark.jittor
def test_replace_sampler_jittordataloader():
    dataset = JittorNormalDataset(20, batch_size=10, shuffle=True)
    dataloader = JittorDataLoader(dataset, batch_size=8, shuffle=True)
    sampler = RandomSampler(dataset)

    replaced_loader = replace_sampler(dataloader, sampler)

    assert not (replaced_loader is dataloader)
    assert not (replaced_loader.dataset.dataset is dataloader.dataset.dataset)
    assert isinstance(replaced_loader.sampler, RandomSampler)
    assert replaced_loader.batch_size == 8
    assert replaced_loader.shuffle == True