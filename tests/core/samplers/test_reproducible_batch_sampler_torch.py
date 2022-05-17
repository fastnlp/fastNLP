from array import array

import pytest

from fastNLP.core.samplers import ReproduceBatchSampler
from fastNLP.core.drivers.torch_driver.utils import replace_batch_sampler
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from tests.helpers.datasets.torch_data import TorchNormalDataset

if _NEED_IMPORT_TORCH:
    import torch
    from torch.utils.data import DataLoader


@pytest.mark.torch
class TestReproducibleBatchSamplerTorch:
    def test_torch_dataloader_1(self):
        # no shuffle
        before_batch_size = 7
        dataset = TorchNormalDataset(num_of_data=100)
        dataloader = DataLoader(dataset, batch_size=before_batch_size)
        re_batchsampler = ReproduceBatchSampler(dataloader.batch_sampler, dataloader.batch_size, drop_last=False)
        dataloader = replace_batch_sampler(dataloader, re_batchsampler)

        forward_steps = 3
        iter_dataloader = iter(dataloader)
        for _ in range(forward_steps):
            next(iter_dataloader)

        # 1. 保存状态
        _get_re_batchsampler = dataloader.batch_sampler
        assert isinstance(_get_re_batchsampler, ReproduceBatchSampler)
        state = _get_re_batchsampler.state_dict()
        assert state == {"index_list": array("I", list(range(100))), "num_consumed_samples": forward_steps*before_batch_size,
                         "sampler_type": "ReproduceBatchSampler"}

        # 2. 断点重训，重新生成一个 dataloader；
        # 不改变 batch_size；
        dataloader = DataLoader(dataset, batch_size=before_batch_size)
        re_batchsampler = ReproduceBatchSampler(dataloader.batch_sampler, dataloader.batch_size, drop_last=False)
        re_batchsampler.load_state_dict(state)
        dataloader = replace_batch_sampler(dataloader, re_batchsampler)

        real_res = []
        supposed_res = (torch.tensor(list(range(21, 28))), torch.tensor(list(range(28, 35))))
        forward_steps = 2
        iter_dataloader = iter(dataloader)
        for _ in range(forward_steps):
            real_res.append(next(iter_dataloader))

        for i in range(forward_steps):
            assert all(real_res[i] == supposed_res[i])

        # 改变 batch_size；
        after_batch_size = 3
        dataloader = DataLoader(dataset, batch_size=after_batch_size)
        re_batchsampler = ReproduceBatchSampler(dataloader.batch_sampler, dataloader.batch_size, drop_last=False)
        re_batchsampler.load_state_dict(state)
        dataloader = replace_batch_sampler(dataloader, re_batchsampler)

        real_res = []
        supposed_res = (torch.tensor(list(range(21, 24))), torch.tensor(list(range(24, 27))))
        forward_steps = 2
        iter_dataloader = iter(dataloader)
        for _ in range(forward_steps):
            real_res.append(next(iter_dataloader))

        for i in range(forward_steps):
            assert all(real_res[i] == supposed_res[i])

        # 断点重训的第二轮是否是一个完整的 dataloader；
        # 先把断点重训所在的那一个 epoch 跑完；
        begin_idx = 27
        while True:
            try:
                data = next(iter_dataloader)
                _batch_size = len(data)
                assert all(data == torch.tensor(list(range(begin_idx, begin_idx + _batch_size))))
                begin_idx += _batch_size
            except StopIteration:
                break

        # 开始新的一轮；
        begin_idx = 0
        iter_dataloader = iter(dataloader)
        while True:
            try:
                data = next(iter_dataloader)
                _batch_size = len(data)
                assert all(data == torch.tensor(list(range(begin_idx, begin_idx + _batch_size))))
                begin_idx += _batch_size
            except StopIteration:
                break

    def test_torch_dataloader_2(self):
        # 测试新的一轮的 index list 是重新生成的，而不是沿用上一轮的；
        from torch.utils.data import DataLoader
        before_batch_size = 7
        dataset = TorchNormalDataset(num_of_data=100)
        # 开启 shuffle，来检验断点重训后的第二轮的 index list 是不是重新生成的；
        dataloader = DataLoader(dataset, batch_size=before_batch_size, shuffle=True)
        re_batchsampler = ReproduceBatchSampler(dataloader.batch_sampler, dataloader.batch_size, drop_last=False)
        dataloader = replace_batch_sampler(dataloader, re_batchsampler)

        # 将一轮的所有数据保存下来，看是否恢复的是正确的；
        all_supposed_data = []
        forward_steps = 3
        iter_dataloader = iter(dataloader)
        for _ in range(forward_steps):
            all_supposed_data.extend(next(iter_dataloader).tolist())

        # 1. 保存状态
        _get_re_batchsampler = dataloader.batch_sampler
        assert isinstance(_get_re_batchsampler, ReproduceBatchSampler)
        state = _get_re_batchsampler.state_dict()

        # 2. 断点重训，重新生成一个 dataloader；
        # 不改变 batch_size；
        dataloader = DataLoader(dataset, batch_size=before_batch_size, shuffle=True)
        re_batchsampler = ReproduceBatchSampler(dataloader.batch_sampler, dataloader.batch_size, drop_last=False)
        re_batchsampler.load_state_dict(state)
        dataloader = replace_batch_sampler(dataloader, re_batchsampler)

        iter_dataloader = iter(dataloader)
        # 先把这一轮的数据过完；
        pre_index_list = dataloader.batch_sampler.state_dict()["index_list"]
        while True:
            try:
                all_supposed_data.extend(next(iter_dataloader).tolist())
            except StopIteration:
                break
        assert all_supposed_data == list(pre_index_list)

        # 重新开启新的一轮；
        for _ in range(3):
            iter_dataloader = iter(dataloader)
            res = []
            while True:
                try:
                    res.extend(next(iter_dataloader).tolist())
                except StopIteration:
                    break
            assert res != all_supposed_data

