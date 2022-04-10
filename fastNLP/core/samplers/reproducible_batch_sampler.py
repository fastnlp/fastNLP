import math
from array import array
from copy import deepcopy
from itertools import chain
from typing import Dict, Union, List

import numpy as np

from fastNLP.core.dataset import DataSet
from fastNLP.core.samplers import ReproducibleIterator






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

        self.index_list = kwargs.get("index_list", self._iterate_sampler())
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
            # 注意 self.index_list 内存放的是全部数据的 index；
            # unsigned long
            _index_lst = array("L", _index_lst)
        else:
            # unsigned int
            _index_lst = array("I", _index_lst)
        return _index_lst

    def __iter__(self):
        if self.need_reinitialize:
            self.index_list = self._iterate_sampler()
            self.data_idx = 0
        else:
            self.need_reinitialize = True

        batch = []
        if self.data_idx:
            index_list = self.index_list[self.data_idx:]
        else:
            index_list = self.index_list
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
            return len(self.index_list) // self.batch_size
        else:
            return (len(self.index_list) + self.batch_size - 1) // self.batch_size

    def state_dict(self) -> Dict:
        return {"index_list": deepcopy(self.index_list), "data_idx": self.data_idx, 'sampler_type': self.__class__.__name__}

    def load_state_dict(self, states: Dict):
        assert states['sampler_type'] == self.__class__.__name__, f"The sampler type in checkpoint is {states['sampler_type']}," \
                                                               f"we cannot use {self.__class__.__name__} to load it."

        _index_list = states["index_list"]
        assert len(_index_list) == len(self.index_list), "The number of samples is different between the checkpoint " \
                                                          "record and current dataset."
        self.index_list = _index_list
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
            return len(self.index_list) // self.batch_size - (len(self.index_list) - self.data_idx) // self.batch_size
        else:
            return (len(self.index_list) + self.batch_size - 1) // self.batch_size - \
                   (len(self.index_list) - self.data_idx + self.batch_size - 1) // self.batch_size


class BucketedBatchSampler(ReproducibleIterator):
    def __init__(self, dataset, length: Union[List[int], str], batch_size:int = 32, num_batch_per_bucket:int = 10,
                 shuffle: bool = True, drop_last: bool = False, seed: int = 0, **kwargs):
        """
        首先按照 sample 的长度排序，然后按照 batch_size*num_batch_per_bucket 为一个桶的大小，sample 只会在这个桶内进行组合，这样
            每个 batch 中的 padding 数量会比较少 （因为桶内的数据的长度都接近）。

        :param dataset: 实现了 __len__ 方法的数据容器。
        :param length: 如果为 List，应当与 dataset 有一样的长度，表示 dataset 中每个元素的数量；仅当传入的 dataset 为 fastNLP 的
            DataSet 时支持传入 str，会将该str理解为 dataset 的 field 名称，若 field 中的元素为 int，则认为该值是 sample 的长度。
            如果否则使用 len() 函数得到每个 sample 中这个 field 的长度。
        :param batch_size: 每个 batch 的大小
        :param num_batch_per_bucket: 多少个 batch 组成一个桶，数据只会在一个桶内进行 shuffle 。
        :param shuffle: 如果为 True，将不进行 shuffle，实际上数据会以从长到短的方式输出。
        :param drop_last: 如果最后一个 batch 的 sample 数量无法凑齐 batch_size 这么多，是否需要丢掉。
        :param seed: 设置的随机数种子
        :param kwargs: fastNLP 保留使用
        """
        super().__init__()
        if not isinstance(dataset, DataSet):
            length = dataset.get_field(length)
            if not isinstance(length[0], int):
                length = list(map(len, length))
        else:
            assert isinstance(length, List) and len(length)==len(dataset), "When the dataset is not fastNLP.DataSet, " \
                                                                           "the length parameter can only be List[int]"
        assert len(length) == len(dataset), "The length of `data` and `length` should be equal."

        if drop_last:
            assert len(dataset)>=batch_size, "The number of samplers must be larger than batch_size when `drop_last=True`."

        self.dataset = dataset
        self.length = np.array(length, dtype=int)  # 按照长到短排列的序号。

        self.batch_size = batch_size
        self.num_batch_per_bucket = num_batch_per_bucket
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self.num_consumed_samples = kwargs.get("num_consumed_samples", 0)  # 总共迭代了多少数据了，包括多卡情况下的其它卡上的输出的数量

        # 多卡的相关的参数
        self.num_replicas = kwargs.get("num_replicas", 1)
        self.rank = kwargs.get("rank", 0)
        self.epoch = kwargs.get("epoch", -1)
        self.pad = kwargs.get("pad", False)  # 该参数在单卡上不具有任何意义；

        # 是否处于iteration之间，为True不允许调用 set_distributed()和load_state_dict()
        self.during_iter = kwargs.get("during_iter", False)

    def set_distributed(self, num_replicas, rank, pad=True):
        assert self.during_iter is False, "Cannot set the sampler to be distributed when it is " \
                                           "during an unfinished iteration."
        assert num_replicas > 0 and isinstance(num_replicas, int)
        assert isinstance(rank, int) and 0 <= rank < num_replicas
        # 注意初始化该函数时，所有的状态都应当默认是一个 epoch 刚开始训练的状态；
        self.num_replicas = num_replicas
        self.rank = rank
        self.pad = pad

        num_samples = (len(self.dataset)+self.num_replicas-1)//self.num_replicas*self.num_replicas if pad \
            else len(self.dataset)

        if self.drop_last:
            assert self.num_replicas*self.batch_size<=num_samples, "The number of samples should be greater " \
                                                                    "than the number of replicates multiplied " \
                                                                    "with batch_size when drop_last=True."

        return self

    @property
    def total_size(self):
        """
        这个变量代表的含义是当前这个sampler会最终产生出的index数量（包括了其它rank的），因为replica和pad的原因，这个值可能等于、
            大于或者小于len(dataset)

        :return:
        """
        return self.num_consumed_samples + self.num_replicas*self.num_left_samples

    @property
    def num_left_samples(self):
        """
        返回当前 iteration 还有多少个 sample 结束，表示的是当前 rank 的还剩多少。

        :return:
        """
        num_consumed_samples = self.num_consumed_samples
        return math.ceil((len(self.dataset) - num_consumed_samples) / self.num_replicas) if \
            self.pad else math.floor(((len(self.dataset) - num_consumed_samples) / self.num_replicas))

    def __len__(self):
        """
        返回当前 sampler 还会返回多少个 batch 的数据

        :return:
        """
        num_sampler_per_rank = self.total_size//self.num_replicas
        num_batches = num_sampler_per_rank//self.batch_size if self.drop_last else \
            (num_sampler_per_rank+self.batch_size-1)//self.batch_size
        return num_batches

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

        # 根据内部的长度进行排序
        sub_length = self.length[indices]  # 取出这个 rank 中的长度
        sorted_indices = np.argsort(sub_length)[::-1]  # 按长度从高到低排序的

        if self.shuffle:
            # 实际的 bucket 大小
            bucket_size = min(len(sorted_indices), self.batch_size * self.num_batch_per_bucket)
            seed = self.seed + self.epoch
            rng = np.random.default_rng(abs(seed))
            num_buckets = (len(sorted_indices) + bucket_size - 1)//bucket_size
            batches = []
            batch_indices = []
            for i in range(num_buckets):
                bucket = sorted_indices[i*bucket_size:(i+1)*bucket_size]
                rng.shuffle(bucket)  # bucket 内部 shuffle 一下
                _indices = np.full(fill_value=self.batch_size, dtype=int,
                                   shape=(len(bucket)//self.batch_size)).cumsum()
                _batches = np.split(bucket, _indices)
                batch_indices.extend(list(range(len(batches), len(batches)+len(_batches))))
                batches.extend(_batches)
            last_batches = []
            if len(batches)>=1 and len(batches[-1])<self.batch_size:
                last_batches = batches[-1].tolist()
                batch_indices = batch_indices[:-1]
                batches = batches[:-1]
            if self.drop_last and len(last_batches)<self.batch_size:
                last_batches = []
            rng.shuffle(batch_indices)  # 不同的 batch 也 shuffle ，当前这种可以保证每张卡上每个 batch 长度都接近的。
            batches = np.array(batches)[batch_indices]
            indices = list(chain(*batches)) + last_batches
        else:
            indices = sorted_indices
            if len(indices)<self.batch_size and self.drop_last:
                indices = []

        for index in range(indices):
            self.num_consumed_samples += self.num_replicas
            yield index
        self.during_iter = False
        self.num_consumed_samples = 0

    def generate_indices(self) -> List[int]:
        """
        生成随机序列，用于保证在所有卡的总和加起来是原来的数据量。

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
        # 如果 self.during_iter 是 True，那么 data_idx 一定是 0；
        assert self.during_iter is False, "Cannot call load_state_dict() when it is " \
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