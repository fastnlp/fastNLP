__all__ = [
    'UnrepeatedSortedSampler',
    'UnrepeatedSampler'
]

from typing import List, Union
from fastNLP.core.dataset import DataSet

import numpy as np


class UnrepeatedSampler:
    def __init__(self, dataset, shuffle: bool = False, seed: int = 0, **kwargs):
        """
        考虑在多卡evaluate的场景下，不能重复sample。

        :param dataset:
        :param shuffle:
        :param seed:
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed

        # 多卡的相关的参数
        self.num_replicas = kwargs.get('num_replicas', 1)
        self.rank = kwargs.get('rank', 0)
        self.epoch = kwargs.get('epoch', -1)

    def __len__(self):
        """
        返回 sampler 一次完整的迭代过程会产生多少个index。多卡的情况下，只考虑当前rank；
        :return:
        """
        num_common = len(self.dataset)//self.num_replicas
        self.num_samples = num_common + int(self.rank < (len(self.dataset)-num_common*self.num_replicas))
        return self.num_samples

    def __iter__(self):
        indices = self.generate_indices()

        # subsample
        indices = indices[self.rank:len(indices):self.num_replicas]
        assert len(indices) == len(self)

        for index in indices:
            yield index

    def generate_indices(self) -> List[int]:
        """
        生成随机序列

        :return:
        """
        if self.shuffle:
            indices = list(range(len(self.dataset)))
            seed = self.seed + self.epoch
            rng = np.random.default_rng(abs(seed))
            rng.shuffle(indices)
            if self.epoch < 0:  # 防止用户忘记调用 set_epoch，至少这样可以保证每次epoch出来的index顺序不同。
                self.epoch -= 1
        else:
            indices = list(range(len(self.dataset)))
        return indices

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_distributed(self, num_replicas, rank):
        """
        该方法本质上等同于 ddp 情形下的没有完成的初始化，应当在初始化该 sampler 本身后立即被调用；

        :param num_replicas:
        :param rank:
        :return:
        """
        assert num_replicas>0 and isinstance(num_replicas, int)
        assert isinstance(rank, int) and 0<=rank<num_replicas
        # 注意初始化该函数时，所有的状态都应当默认是一个 epoch 刚开始训练的状态；
        self.num_replicas = num_replicas
        self.rank = rank

        return self


class UnrepeatedSortedSampler(UnrepeatedSampler):
    def __init__(self, dataset, length:Union[str, List], seed: int = 0):
        """
        将 dataset 中的数据根据 length 从长到短进行迭代，并且保证在多卡场景下数据不重复。本 sampler 可能导致各个机器上的
            batch 数量不完全一致。

        :param dataset: 实现了 __len__ 方法的数据容器。
        :param length: 如果为 List，应当与 dataset 有一样的长度，表示 dataset 中每个元素的数量；仅当传入的 dataset 为 fastNLP 的
            DataSet 时支持传入 str，会将该str理解为 dataset 的 field 名称，若 field 中的元素为 int，则认为该值是 sample 的长度。
        :param shuffle: 如果为 True，将不进行 shuffle，实际上数据会以从长到短的方式输出。
        :param seed: 设置的随机数种子
        :param kwargs: fastNLP 保留使用
        """
        super().__init__(dataset=dataset, shuffle=False, seed=seed)
        if isinstance(dataset, DataSet):
            length = dataset.get_field(length)
            if not isinstance(length[0], int):
                length = list(map(len, length))
        else:
            assert len(length) == len(dataset), "When the dataset is not fastNLP.DataSet, " \
                                                "the length parameter can only be List[int]"

        assert len(length) == len(dataset), "The length of `data` and `length` should be equal."

        self.length = np.array(length, dtype=int)  # 按照长到短排列的序号。
        self.sorted_indices = np.argsort(self.length)[::-1].tolist()  # 按长度从高到低排序的

    def generate_indices(self) -> List[int]:
        return self.sorted_indices
