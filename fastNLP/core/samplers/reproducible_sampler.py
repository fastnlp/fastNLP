__all__ = [
    'ReproducibleSampler',
    'RandomSampler',
    "SortedSampler",
    "SequentialSampler"
]

from typing import Dict, List, Union
import math

import numpy as np

from fastNLP.core.log import logger
from fastNLP.core.dataset import DataSet


class ReproducibleSampler:
    """
    可复现的 Sampler 对象。

    注意所有继承 `ReproducibleSampler` 的类的  `__init__` 方法中都需要加入参数 `**kwargs`，用来使我们再断点重训时重新实例化这个 sampler
     或者 batch_sampler；注意，所有在 init 中初始化的变量，都不能含有 _ 下横线作为开头；所有不在 init 中设置的变量都必须以下横线开头。

    """
    def __init__(self, **kwargs):
        self.num_replicas = 1

    def set_distributed(self, num_replicas, rank, pad=True):
        raise NotImplementedError("Each specific sampler should implement its own `set_distributed` method.")

    def __len__(self):
        raise NotImplementedError("Each specific sampler should implement its own `__len__` method.")

    def __iter__(self):
        raise NotImplementedError("Each specific sampler should implement its own `__iter__` method.")

    def state_dict(self):
        """

        :return:
        """
        raise NotImplementedError("Each specific sampler should implement its own `state_dict` method.")

    def load_state_dict(self, states):
        raise NotImplementedError("Each specific sampler should implement its own `load_state_dict` method.")

    @property
    def num_left_samples(self):
        raise NotImplementedError("Each specific sampler should implement its own `num_left_samples` method.")

    def set_epoch(self, epoch):
        pass


class RandomSampler(ReproducibleSampler):
    def __init__(self, dataset, shuffle: bool = True, seed: int = 0, **kwargs):
        """

        :param dataset: 实现了 __len__ 方法的数据容器
        :param shuffle: 是否在每次 iterate 的时候打乱顺序。
        :param seed: 随机数种子。
        :param kwargs: 用户不需要使用，fastNLP 内部使用
        """
        super(RandomSampler, self).__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        
        self.num_consumed_samples = kwargs.get("num_consumed_samples", 0)  # 总共迭代了多少数据了，包括多卡情况下的其它卡上的输出的数量

        # 多卡的相关的参数
        self.num_replicas = kwargs.get("num_replicas", 1)
        self.rank = kwargs.get("rank", 0)
        self.epoch = kwargs.get("epoch", -1)
        self.pad = kwargs.get("pad", False)  # 该参数在单卡上不具有任何意义；

        # 是否处于iteration之间，为True不允许调用 set_distributed()和load_state_dict()
        self.during_iter = kwargs.get("during_iter", False)

    def __len__(self):
        """
        返回 sampler 一次完整的迭代过程会产生多少个index。多卡的情况下，只考虑当前rank；
        :return:
        """
        return self.total_size//self.num_replicas

    def __iter__(self):
        r"""
        当前使用num_consumed_samples做法会在交替使用的时候遇到问题；
        Example:
            >>> sampler = RandomSampler()
            >>> iter1 = iter(sampler)
            >>> iter2 = iter(sampler)
            >>> next(iter1)
            >>> next(iter2)  # 当前num_consumed_samples的数量会发生变化
        """

        if self.during_iter:  # 如果发现_during_iter为True，说明之前的还没结束，只有强制重新初始化了
            self.num_consumed_samples = 0
        self.during_iter = True
        indices = self.generate_indices()

        if self.pad:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.num_consumed_samples:]
        indices = indices[self.rank:len(indices):self.num_replicas]
        assert len(indices) == self.num_left_samples
        for idx, index in enumerate(indices, start=1):
            self.num_consumed_samples += self.num_replicas
            yield index
        self.during_iter = False
        self.num_consumed_samples = 0

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

    def state_dict(self) -> Dict:
        states = {'seed': self.seed, 'epoch': self.epoch, 'num_consumed_samples': self.num_consumed_samples,
                  'sampler_type': self.__class__.__name__, 'length': len(self.dataset), 'shuffle': self.shuffle}
        return states

    def load_state_dict(self, states: Dict):
        # 如果 self.during_iter 是 True，那么 num_consumed_samples 一定是 0；
        assert self.during_iter is False, "Cannot call load_state_dict() when it is " \
                                           "during an unfinished iteration."

        assert states['sampler_type'] == self.__class__.__name__, f"The sampler type in checkpoint is {states['sampler_type']}," \
                                                                  f"we cannot use {self.__class__.__name__} to load it."

        length = states['length']
        assert length == len(self.dataset), f"The number of samples is different between the checkpoint record({length}) " \
                                            f"and current dataset({len(self.dataset)})."
        self.seed = states['seed']
        self.epoch = states['epoch']
        self.num_consumed_samples = states['num_consumed_samples']
        if self.num_consumed_samples >= length:  # 如果保存的时候已经到达了最后一个sample了，则直接将结果重置为0
            self.num_consumed_samples = 0
        if self.shuffle != states['shuffle']:
            logger.info(f"The shuffle from the checkpoint is {states['shuffle']}, while set as {self.shuffle}, "
                        f"we use shuffle={states['shuffle']}")
        self.shuffle = states["shuffle"]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_distributed(self, num_replicas, rank, pad=True):
        """
        该方法本质上等同于 ddp 情形下的没有完成的初始化，应当在初始化该 sampler 本身后立即被调用；

        :param num_replicas:
        :param rank:
        :param pad: 这个 pad 的意思是指如果 sample 数量不整除 num_replicas 的时候，要不要 pad 一下，使得最终使得 replica 上
            的 sample 数量是完全一致的。
        :return:
        """

        assert self.during_iter is False, "Cannot set the sampler to be distributed when it is " \
                                           "during an unfinished iteration."
        assert num_replicas>0 and isinstance(num_replicas, int)
        assert isinstance(rank, int) and 0<=rank<num_replicas
        # 注意初始化该函数时，所有的状态都应当默认是一个 epoch 刚开始训练的状态；
        self.num_replicas = num_replicas
        self.rank = rank
        self.pad = pad

        return self

    @property
    def total_size(self):
        """
        这个变量代表的含义是当前这个sampler会最终产生出的index数量，因为replica和pad的原因，这个值可能等于、大于或者小于len(dataset)

        :return:
        """
        return self.num_consumed_samples + self.num_replicas*self.num_left_samples

    @property
    def num_left_samples(self):
        """
        返回当前 iteration 还有多少个 sample 结束。表示的是当前 rank 的还剩多少

        :return:
        """
        num_consumed_samples = self.num_consumed_samples
        return math.ceil((len(self.dataset) - num_consumed_samples) / self.num_replicas) if \
            self.pad else math.floor(((len(self.dataset) - num_consumed_samples) / self.num_replicas))


class SequentialSampler(RandomSampler):
    def __init__(self, dataset, **kwargs):
        """
        按照顺序读取 dataset 。在多卡情况下，间隔读取，例如，在两卡情况下，卡0取 [0,2,4,..], 卡1取 [1,3,5...]。

        :param dataset: 实现了 __len__ 方法的数据容器。
        :param kwargs:
        """
        super().__init__(dataset=dataset, **kwargs)

    def __iter__(self):
        if self.during_iter:  # 如果发现_during_iter为True，说明之前的还没结束，只有强制重新初始化了
            self.num_consumed_samples = 0
        self.during_iter = True
        indices = self.generate_indices()

        if self.pad:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.num_consumed_samples:]
        indices = indices[self.rank:len(indices):self.num_replicas]
        assert len(indices) == self.num_left_samples

        for idx, index in enumerate(indices, start=1):
            self.num_consumed_samples += self.num_replicas
            yield index
        self.during_iter = False
        self.num_consumed_samples = 0

    def generate_indices(self) -> List[int]:
        """
        生成随机序列

        :return:
        """
        return list(range(len(self.dataset)))

    def state_dict(self) -> Dict:
        states = {'num_consumed_samples': self.num_consumed_samples, 'sampler_type': self.__class__.__name__,
                  'length': len(self.dataset)
                  }
        return states

    def load_state_dict(self, states: Dict):
        # 如果 self.during_iter 是 True，那么 num_consumed_samples 一定是 0；
        assert self.during_iter is False, "Cannot call load_state_dict() when it is " \
                                          "during an unfinished iteration."

        assert states['sampler_type'] == self.__class__.__name__, f"The sampler type in checkpoint is {states['sampler_type']}," \
                                                                  f"we cannot use {self.__class__.__name__} to load it."

        length = states['length']
        assert length == len(self.dataset), f"The number of samples is different between the checkpoint record({length}) " \
                                            f"and current dataset({len(self.dataset)})."
        self.num_consumed_samples = states['num_consumed_samples']
        if self.num_consumed_samples >= length:  # 如果保存的时候已经到达了最后一个sample了，则直接将结果重置为0
            self.num_consumed_samples = 0


class SortedSampler(SequentialSampler):
    def __init__(self, dataset, length:Union[str, List], **kwargs):
        """
        将 dataset 中的数据根据 length 从长到短进行迭代。在多卡情况下，由于padding 最后一个 sample 可能是最长的那个 sample。

        :param dataset: 实现了 __len__ 方法的数据容器。
        :param length: 如果为 List，应当与 dataset 有一样的长度，表示 dataset 中每个元素的数量；仅当传入的 dataset 为 fastNLP 的
            DataSet 时支持传入 str，会将该str理解为 dataset 的 field 名称，若 field 中的元素为 int，则认为该值是 sample 的长度。
        :param seed: 设置的随机数种子
        :param kwargs: fastNLP 保留使用
        """
        super().__init__(dataset=dataset, **kwargs)
        if isinstance(dataset, DataSet) and isinstance(length, str):
            length = dataset.get_field(length).content
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

    def __iter__(self):
        if self.during_iter:  # 如果发现_during_iter为True，说明之前的还没结束，只有强制重新初始化了
            self.num_consumed_samples = 0
        self.during_iter = True
        indices = self.generate_indices()

        if self.pad:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.num_consumed_samples:]
        indices = indices[self.rank:len(indices):self.num_replicas]
        assert len(indices) == self.num_left_samples

        for idx, index in enumerate(indices, start=1):
            self.num_consumed_samples += self.num_replicas
            yield index
        self.during_iter = False
        self.num_consumed_samples = 0

