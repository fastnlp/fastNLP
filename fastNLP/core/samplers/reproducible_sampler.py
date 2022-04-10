from typing import Dict, List
import math
import numpy as np
from array import array
from copy import deepcopy


__all__ = [
    'ReproducibleIterator',
    'RandomSampler',
    'ReproducibleBatchSampler',
    're_instantiate_sampler'
]


def re_instantiate_sampler(sampler):
    all_attributes = vars(sampler)
    return type(sampler)(**all_attributes)



class ReproducibleIterator:
    """
    注意所有继承 `ReproducibleIterator` 的类的  `__init__` 方法中都需要加入参数 `**kwargs`，用来使我们再断点重训时重新实例化这个 sampler
     或者 batch_sampler；
    """

    def set_distributed(self, num_replicas, rank, pad=True):
        raise NotImplementedError("Each specific sampler should implement its own `set_distributed` method.")

    def __len__(self):
        raise NotImplementedError("Each specific sampler should implement its own `__len__` method.")

    def __iter__(self):
        raise NotImplementedError("Each specific sampler should implement its own `__iter__` method.")

    def state_dict(self):
        raise NotImplementedError("Each specific sampler should implement its own `state_dict` method.")

    def load_state_dict(self, states):
        raise NotImplementedError("Each specific sampler should implement its own `load_state_dict` method.")

    @property
    def num_left_samples(self):
        raise NotImplementedError("Each specific sampler should implement its own `num_left_samples` method.")

    def set_epoch(self, epoch):
        pass


class RandomSampler(ReproducibleIterator):
    def __init__(self, dataset, shuffle: bool = True, seed: int = 0, **kwargs):
        """


        :param dataset: 实现了 __len__ 方法的数据容器
        :param shuffle: 是否在每次 iterate 的时候打乱顺序。
        :param seed: 随机数种子。
        :param kwargs: 用户不需要使用，fastNLP 内部使用
        """

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
        self._during_iter = kwargs.get("_during_iter", False)

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

        if self._during_iter:  # 如果发现_during_iter为True，说明之前的还没结束，只有强制重新初始化了
            self.num_consumed_samples = 0
        self._during_iter = True
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

        for index in indices:
            self.num_consumed_samples += self.num_replicas
            yield index
        self._during_iter = False
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
        states = {
            'seed': self.seed,
            'epoch': self.epoch,
            'num_consumed_samples': self.num_consumed_samples,  # 注意该值是计算所有 rank 上训练的所有数据；
            'sampler_type': self.__class__.__name__,
            'length': len(self.dataset),
            'shuffle': self.shuffle
        }
        return states

    def load_state_dict(self, states: Dict):
        # 如果 self._during_iter 是 True，那么 data_idx 一定是 0；
        assert self._during_iter is False, "Cannot call load_state_dict() when it is " \
                                           "during an unfinished iteration."

        assert states['sampler_type'] == self.__class__.__name__, f"The sampler type in checkpoint is {states['sampler_type']}," \
                                                                  f"we cannot use {self.__class__.__name__} to load it."

        length = states['length']
        assert length == len(self.dataset), "The number of samples is different between the checkpoint record " \
                                            "and current dataset."
        self.seed = states['seed']
        self.epoch = states['epoch']
        self.num_consumed_samples = states['num_consumed_samples']
        if self.num_consumed_samples>=length:  # 如果保存的时候已经到达了最后一个sample了，则直接将结果重置为0
            self.num_consumed_samples = 0
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

        assert self._during_iter is False, "Cannot set the sampler to be distributed when it is " \
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
        返回当前 iteration 还有多少个 sample 结束

        :return:
        """
        num_consumed_samples = self.num_consumed_samples
        return math.ceil((len(self.dataset) - num_consumed_samples) / self.num_replicas) if \
            self.pad else math.floor(((len(self.dataset) - num_consumed_samples) / self.num_replicas))


class ReproducibleBatchSampler:
    # 这两个参数的值应当交给 driver 的 get_dataloader_args 函数去拿；
    def __init__(self, batch_sampler, batch_size: int, drop_last: bool, **kwargs):
        """
        可以使得 batch_sampler 对象状态恢复的 wrapper 。

        :param batch_sampler: 可迭代出 数字 或 数字列表 的可迭代对象。ReproducibleBatchSampler 将首先遍历一边该对象，然后将迭代
            出来的序号暂存起来，使用时按照 batch_size 的 batch 大小吐出序号列表。
        :param batch_size: 每个 batch 的大小是多少。
        :param drop_last: 如果最后一个 batch 无法构成 batch_size 那么多个 sample ，是否丢掉。
        :param kwargs: fastNLP 内部使用。
        """
        self.batch_sampler = batch_sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.data_idx = kwargs.get("data_idx", 0)

        self._index_list = kwargs.get("_index_list", self._iterate_sampler())
        self.need_reinitialize = kwargs.get("need_reinitialize", False)

    def _iterate_sampler(self):
        _index_lst = []
        for idx in self.batch_sampler:
            if isinstance(idx, list):
                _index_lst.extend(idx)
            # 说明是在初始化时传入的是一个 sampler，理论上对应于 dataloader 在初始化时没有 batch_size，也没有 batch_sampler 的情况；
            else:
                _index_lst.append(idx)
        # 64 位机器的 unsigned int 为 4 个字节，能表示的最大大小为 4294967295；
        if len(_index_lst) > 4294967295:
            # 注意 self._index_list 内存放的是全部数据的 index；
            # unsigned long
            _index_lst = array("L", _index_lst)
        else:
            # unsigned int
            _index_lst = array("I", _index_lst)
        return _index_lst

    def __iter__(self):
        if self.need_reinitialize:
            self._index_list = self._iterate_sampler()
            self.data_idx = 0
        else:
            self.need_reinitialize = True

        batch = []
        if self.data_idx:
            index_list = self._index_list[self.data_idx:]
        else:
            index_list = self._index_list
        for idx in index_list:
            batch.append(idx)
            self.data_idx += 1
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self._index_list) // self.batch_size
        else:
            return (len(self._index_list) + self.batch_size - 1) // self.batch_size

    def state_dict(self) -> Dict:
        return {"index_list": deepcopy(self._index_list), "data_idx": self.data_idx, 'sampler_type': self.__class__.__name__}

    def load_state_dict(self, states: Dict):
        assert states['sampler_type'] == self.__class__.__name__, f"The sampler type in checkpoint is {states['sampler_type']}," \
                                                               f"we cannot use {self.__class__.__name__} to load it."

        _index_list = states["index_list"]
        assert len(_index_list) == len(self._index_list), "The number of samples is different between the checkpoint " \
                                                          "record and current dataset."
        self._index_list = _index_list
        self.data_idx = states["data_idx"]
        self.need_reinitialize = False

    def set_distributed(self):
        raise RuntimeError(f"ReproduceBatchSampler does not support to change to distributed training.")

    def set_epoch(self, epoch):
        if hasattr(self.batch_sampler, "sampler") and hasattr(self.batch_sampler.sampler, 'set_epoch') and callable(self.batch_sampler.sampler.set_epoch):
            self.batch_sampler.sampler.set_epoch(epoch)

    @property
    def batch_idx_in_epoch(self):
        if self.drop_last:
            return len(self._index_list) // self.batch_size - (len(self._index_list) - self.data_idx) // self.batch_size
        else:
            return (len(self._index_list) + self.batch_size - 1) // self.batch_size - \
                (len(self._index_list) - self.data_idx + self.batch_size - 1) // self.batch_size

# todo
# class SortedSampler(ReproducibleIterator):
#     def __init__(self, dataset, key):
#         pass
#
#
# class BucketedSampler(ReproducibleIterator):
#     def __init__(self, dataset, key):
#         pass

if __name__ == "__main__":

    sampler = RandomSampler(1)

    print(vars(sampler))

    batch_sampler = ReproducibleBatchSampler(list(range(3)), 1, True)
    print(vars(batch_sampler))




