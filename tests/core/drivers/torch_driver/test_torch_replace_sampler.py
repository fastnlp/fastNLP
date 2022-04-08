
import pytest


from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch
    from torch.utils.data import DataLoader


# 框架无关的一些接口测试

"""
模拟
同一个dl，同时传入trainer和evaluator，
    （1）在训练到一半进行evaluate，需要保证trainer中dl的sampler状态不受影响
    （2）evaluate设置新的set_distributed不改变原有trainer中的evaluate
"""

class SequenceDataSet:
    def __init__(self, num_samples):
        self.data = list(range(num_samples))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def check_replace_sampler(driver):
    # dist_sampler 可以选择的有['dist', 'unrepeatdist', None]或者是ReproducibleSampler，ReproducibleBatchSampler
    # reproducible 是 True 和 False

    assert driver.is_distributed() is False, "This test only for non distributed sampler."
    ds = SequenceDataSet(10)
    dataloader = DataLoader(dataset=ds, batch_size=2, collate_fn=lambda x:x, shuffle=True)

    dl1 = driver.replace_sampler(dataloader, dist_sampler='dist', reproducible=True)

    # 迭代两个 batch
    already_seen_idx = set()
    for idx, batch in enumerate(dl1):
        already_seen_idx.update(batch)
        if idx > 1:
            sampler_states = dataloader.sampler.state_dict()
            break

    # 再对原来的dataloader进行迭代，应该不影响 dl1 ，即 dl1 应该继续输出剩下的，而不会重复
    for idx, batch in enumerate(dataloader):
        pass

    left_idxes = set()
    for idx, batch in enumerate(dl1):
        for b in batch:
            assert b not in already_seen_idx
        left_idxes.update(batch)

    if not driver.is_distributed():
        # 如果不是分布式的话，应该是等于整个数据的
        assert len(left_idxes)+len(already_seen_idx) == len(ds)

    # 重新加载，应该是可以输出刚才完全一样的
    dl1.sampler.load_state_dict(sampler_states)
    for idx, batch in enumerate(dl1):
        for b in batch:
            assert b not in already_seen_idx
            assert b in left_idxes











