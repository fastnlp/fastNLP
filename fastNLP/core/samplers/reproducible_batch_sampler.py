__all__ = [
    'BucketedBatchSampler',
    "ReproduceBatchSampler",
    "RandomBatchSampler"
]

import math
from copy import deepcopy
from typing import Dict, Union, List
from itertools import chain
import os

import numpy as np

from fastNLP.core.dataset import DataSet
from fastNLP.core.log import logger
from .utils import create_array
from abc import abstractmethod


class ReproducibleBatchSampler:
    def __init__(self, **kwargs):
        self.num_replicas = 1

    @abstractmethod
    def set_distributed(self, num_replicas, rank, pad=True):
        raise NotImplementedError("Each specific batch_sampler should implement its own `set_distributed` method.")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Each specific batch_sampler should implement its own `__len__` method.")

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError("Each specific batch_sampler should implement its own `__iter__` method.")

    @abstractmethod
    def state_dict(self):
        """

        :return:
        """
        raise NotImplementedError("Each specific batch_sampler should implement its own `state_dict` method.")

    @abstractmethod
    def load_state_dict(self, states):
        raise NotImplementedError("Each specific batch_sampler should implement its own `load_state_dict` method.")

    @abstractmethod
    def set_epoch(self, epoch):
        pass

    @property
    def batch_idx_in_epoch(self):
        raise NotImplementedError("Each specific batch_sampler should implement its own `batch_idx_in_epoch` property.")


class ReproduceBatchSampler(ReproducibleBatchSampler):
    # 这两个参数的值应当交给 driver 的 get_dataloader_args 函数去拿；
    def __init__(self, batch_sampler, batch_size: int, drop_last: bool, **kwargs):
        """
        可以使得 batch_sampler 对象状态恢复的 wrapper 。

        :param batch_sampler: 可迭代出 数字 或 数字列表 的可迭代对象。ReproduceBatchSampler 将首先遍历一边该对象，然后将迭代
            出来的序号暂存起来，使用时按照 batch_size 的 batch 大小吐出序号列表。
        :param batch_size: 每个 batch 的大小是多少。
        :param drop_last: 如果最后一个 batch 无法构成 batch_size 那么多个 sample ，是否丢掉。
        :param kwargs: fastNLP 内部使用。
        """
        super().__init__()

        self.batch_sampler = batch_sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.num_consumed_samples = kwargs.get("num_consumed_samples", 0)

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
        _index_lst = create_array(len(_index_lst), _index_lst)
        return _index_lst

    def __iter__(self):
        if self.need_reinitialize:
            self.index_list = self._iterate_sampler()
            self.num_consumed_samples = 0
        else:
            self.need_reinitialize = True

        batch = []
        if self.num_consumed_samples:
            index_list = self.index_list[self.num_consumed_samples:]
        else:
            index_list = self.index_list

        # 暂时弃用。记住每个 batch 对应的 consumed_samples, 需要这个原因是由于现在的 dataloader 都存在预取数据的设计，需要再结合Trainer中
        #  batch_idx_in_epoch 才能最终确定实际消耗的数据。这个变量需要记录每次yield出去时的真实 num_consumed_samples 的数值。
        # self.num_consumed_samples_array = NumConsumedSamplesArray(buffer_size=os.environ.get(FASTNLP_DEQUE_SIZE, 30),
        #                                                          num_consumed_samples=self.num_consumed_samples)
        for idx in index_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                self.num_consumed_samples += self.batch_size  # [16, 32, 48, 64,..., ]
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            self.num_consumed_samples += len(batch)
            yield batch
        # 需要重置防止边界条件问题
        self.num_consumed_samples = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.index_list) // self.batch_size
        else:
            return (len(self.index_list) + self.batch_size - 1) // self.batch_size

    def state_dict(self) -> Dict:
        states = {
            "index_list": deepcopy(self.index_list),
            "num_consumed_samples": self.num_consumed_samples,
            'sampler_type': self.__class__.__name__
        }
        return states

    def load_state_dict(self, states: Dict):
        assert states['sampler_type'] == self.__class__.__name__, f"The sampler type in checkpoint is {states['sampler_type']}," \
                                                               f"we cannot use {self.__class__.__name__} to load it."

        _index_list = states["index_list"]
        assert len(_index_list) == len(self.index_list), "The number of samples is different between the checkpoint " \
                                                          "record and current dataset."
        self.index_list = _index_list
        self.num_consumed_samples = states["num_consumed_samples"]
        self.need_reinitialize = False

    def set_distributed(self, num_replicas, rank, pad=True):
        raise RuntimeError(f"ReproduceBatchSampler does not support to change to distributed training.")

    def set_epoch(self, epoch):
        if hasattr(self.batch_sampler, "sampler") and hasattr(self.batch_sampler.sampler, 'set_epoch') and callable(self.batch_sampler.sampler.set_epoch):
            self.batch_sampler.sampler.set_epoch(epoch)

    @property
    def batch_idx_in_epoch(self):
        if self.drop_last:
            return len(self.index_list) // self.batch_size - (len(self.index_list) - self.num_consumed_samples) // self.batch_size
        else:
            return (len(self.index_list) + self.batch_size - 1) // self.batch_size - \
                   (len(self.index_list) - self.num_consumed_samples + self.batch_size - 1) // self.batch_size


class RandomBatchSampler(ReproducibleBatchSampler):
    def __init__(self, dataset, batch_size:int = 32, shuffle: bool = True,
                 drop_last: bool = False, seed: int = 0, **kwargs):
        """
        随机分 batch 的 batch_sampler 。

        :param dataset: 实现了 __len__ 方法的数据容器。
        :param batch_size: 每个 batch 的大小
        :param shuffle: 如果为 True，将不进行 shuffle，实际上数据会以从长到短的方式输出。
        :param drop_last: 如果最后一个 batch 的 sample 数量无法凑齐 batch_size 这么多，是否需要丢掉。
        :param seed: 设置的随机数种子
        :param kwargs: fastNLP 保留使用
        """
        super().__init__()

        self.dataset = dataset

        self.batch_size = batch_size
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

        # 以下变量为内部使用恢复状态的变量。
        self.old_batch_size = kwargs.get('old_batch_size', self.batch_size)

    def set_distributed(self, num_replicas, rank, pad=True):
        assert self.during_iter is False, "Cannot set the sampler to be distributed when it is " \
                                          "during an unfinished iteration."
        assert num_replicas > 0 and isinstance(num_replicas, int)
        assert isinstance(rank, int) and 0 <= rank < num_replicas
        # 注意初始化该函数时，所有的状态都应当默认是一个 epoch 刚开始训练的状态；
        self.num_replicas = num_replicas
        self.rank = rank
        self.pad = pad

        return self

    def __iter__(self):
        if self.during_iter:  # 如果发现_during_iter为True，说明之前的还没结束，只有强制重新初始化了
            self.num_consumed_samples = 0
        self.during_iter = True

        indices = list(range(len(self.dataset)))

        if self.shuffle:
            if self.num_consumed_samples > 0:  # 需要先按照原来的排序，删掉多余的
                _batches = []
                for _i in range(self.old_num_replicas):
                    _indices = indices[_i:len(indices):self.old_num_replicas]
                    __batches = self.batchify(_indices, self.old_batch_size, seed=self.seed + self.epoch)
                    _batches.append(__batches)
                batches = list(chain(*[_ for _ in zip(*_batches)]))
                indices = list(chain(*batches))
                indices = indices[self.num_consumed_samples:]
            # 取出这个 rank ，
            indices = indices[self.rank:len(indices):self.num_replicas]
            batches = self.batchify(indices, self.batch_size, seed=self.seed + self.epoch)
            batches = list(map(list, batches))
        else:
            indices = indices[self.num_consumed_samples:]
            indices = indices[self.rank:len(indices):self.num_replicas]
            _num_batches = len(indices) // self.batch_size
            if _num_batches == 0:
                batches = [indices]
            else:
                batches = list(map(list, np.array_split(indices[:_num_batches*self.batch_size], _num_batches)))
                if len(indices)%self.batch_size!=0:
                    batches.append(indices[_num_batches*self.batch_size:])

        need_pad_num = (len(self.dataset)-self.num_consumed_samples) % self.num_replicas
        if self.pad and need_pad_num !=0 and need_pad_num<=self.rank:
            if len(batches) > 0:
                if len(batches[-1])<self.batch_size:
                    batches[-1].append(batches[-1][0])  # 这里可以保证这个bucket的长度没被破坏。
                else:
                    batches.append([batches[-1][0]])
        elif self.pad is False and need_pad_num !=0 and need_pad_num>self.rank:
            if len(batches):
                batches[-1].pop(-1)
            if len(batches[-1])==0:
                batches.pop(-1)

        assert sum(map(len, batches)) == self.num_left_samples

        if self.drop_last and len(batches) >= 1 and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        for batch in batches:
            self.num_consumed_samples += self.num_replicas * len(batch)
            yield list(map(int, batch))
        self.during_iter = False
        self.num_consumed_samples = 0
        self.old_batch_size = self.batch_size
        self.old_num_replicas = self.num_replicas
        if self.epoch < 0:  # 防止用户没有修改epoch，导致每个epoch都一样了
            self.epoch -= 1

    def batchify(self, indices, batch_size, seed):
        """
        将 indices 分为 batches

        :param sorted_indices: List[int]
        :param batch_size: int
        :param seed: int
        :return:  List[List[int]]
        """
        # 实际的 bucket 大小
        rng = np.random.default_rng(abs(seed))
        rng.shuffle(indices)
        num_samples = 0
        batches = []
        while num_samples<len(indices):
            batches.append(indices[num_samples:num_samples+batch_size])
            num_samples += batch_size
        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    @property
    def batch_idx_in_epoch(self):
        if self.drop_last:
            return len(self.dataset) // self.num_replicas // self.batch_size - self.num_left_samples // self.batch_size
        else:
            return (len(self.dataset) // self.num_replicas + self.batch_size - 1) // self.batch_size - \
                   (self.num_left_samples + self.batch_size - 1) // self.batch_size

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

    def __len__(self)->int:
        """
        返回当前 sampler 还会返回多少个 batch 的数据

        :return:
        """
        num_sampler_per_rank = self.total_size//self.num_replicas
        num_batches = num_sampler_per_rank//self.batch_size if self.drop_last else \
            (num_sampler_per_rank+self.batch_size-1)//self.batch_size
        return num_batches

    def state_dict(self) -> Dict:
        if self.old_batch_size != self.batch_size:
            raise RuntimeError("BucketedBatchSampler does not support saving before last checkpoint states have been"
                               " consumed. ")
        states = {'seed': self.seed, 'epoch': self.epoch, 'num_consumed_samples': self.num_consumed_samples,
                  'sampler_type': self.__class__.__name__, 'length': len(self.dataset), 'shuffle': self.shuffle,
                  'batch_size': self.batch_size,
                  'num_replicas': self.num_replicas}

        return states

    def load_state_dict(self, states: Dict):
        # 如果 self.during_iter 是 True，那么 num_consumed_samples 一定是 0；
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
        if self.shuffle != states['shuffle']:
            logger.info(f"The shuffle from the checkpoint is {states['shuffle']}, while set as {self.shuffle}, "
                        f"we use shuffle={states['shuffle']}")
        self.shuffle = states["shuffle"]
        self.old_batch_size = states['batch_size']
        self.old_num_replicas = states['num_replicas']


class BucketedBatchSampler(ReproducibleBatchSampler):
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
        if isinstance(dataset, DataSet) and isinstance(length, str):
            length = dataset.get_field(length).content
            if not isinstance(length[0], int):
                length = list(map(len, length))
        else:
            assert len(length) == len(dataset), "When the dataset is not fastNLP.DataSet, " \
                                              "the length parameter can only be List[int]"

        assert len(length) == len(dataset), "The length of `data` and `length` should be equal."

        self.dataset = dataset
        self.length = np.array(length, dtype=int)  # 按照长到短排列的序号。
        self.sorted_indices = np.argsort(self.length)[::-1]  # 按长度从高到低排序的

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

        # 以下变量为内部使用恢复状态的变量。
        self.old_batch_size = kwargs.get('old_batch_size', self.batch_size)
        self.old_num_batch_per_bucket = kwargs.get('old_num_batch_per_bucket', self.num_batch_per_bucket)

    def set_distributed(self, num_replicas, rank, pad=True):
        assert self.during_iter is False, "Cannot set the sampler to be distributed when it is " \
                                           "during an unfinished iteration."
        assert num_replicas > 0 and isinstance(num_replicas, int)
        assert isinstance(rank, int) and 0 <= rank < num_replicas
        # 注意初始化该函数时，所有的状态都应当默认是一个 epoch 刚开始训练的状态；
        self.num_replicas = num_replicas
        self.rank = rank
        self.pad = pad

        # num_samples = (len(self.dataset)+self.num_replicas-1)//self.num_replicas*self.num_replicas if pad \
        #     else len(self.dataset)
        #
        # if self.drop_last:
        #     assert self.num_replicas*self.batch_size<=num_samples, "The number of samples should be greater " \
        #                                                             "than the number of replicates multiplied " \
        #                                                             "with batch_size when drop_last=True."

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

    def __len__(self)->int:
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

        sorted_indices = deepcopy(self.sorted_indices).tolist()  # 按长度从高到低排序的

        if self.shuffle:
            if self.num_consumed_samples > 0:  # 需要先按照原来的排序，删掉多余的
                _batches = []
                for _i in range(self.old_num_replicas):
                    _sorted_indices = sorted_indices[_i:len(sorted_indices):self.old_num_replicas]
                    __batches = self.bucketerize(_sorted_indices, self.old_batch_size, self.old_num_batch_per_bucket,
                                               seed=self.seed+self.epoch)
                    _batches.append(__batches)
                batches = list(chain(*[_ for _ in zip(*_batches)]))
                sorted_indices = list(chain(*batches))
                sorted_indices = sorted_indices[self.num_consumed_samples:]
                # 再进行排序
                sub_length = self.length[sorted_indices]
                sorted_indices = np.array(sorted_indices)[np.argsort(sub_length)[::-1]]  # 按长度从高到低排序的
            # 取出这个 rank ，
            sorted_indices = sorted_indices[self.rank:len(sorted_indices):self.num_replicas]
            batches = self.bucketerize(sorted_indices, self.batch_size, self.num_batch_per_bucket,
                                       seed=self.seed+self.epoch)
            batches = list(map(list, batches))
        else:
            sorted_indices = sorted_indices[self.num_consumed_samples:]
            sorted_indices = sorted_indices[self.rank:len(sorted_indices):self.num_replicas]
            _num_batches = len(sorted_indices) // self.batch_size
            if _num_batches == 0:
                batches = [sorted_indices]
            else:
                batches = list(map(list, np.array_split(sorted_indices[:_num_batches*self.batch_size], _num_batches)))
                if len(sorted_indices)%self.batch_size!=0:
                    batches.append(sorted_indices[_num_batches*self.batch_size:])

        need_pad_num = (len(self.dataset)-self.num_consumed_samples) % self.num_replicas
        if self.pad and need_pad_num !=0 and need_pad_num<=self.rank:
            if len(batches) > 0:
                if len(batches[-1])<self.batch_size:
                    batches[-1].append(batches[-1][0])  # 这里可以保证这个bucket的长度没被破坏。
                else:
                    batches.append([batches[-1][0]])
        elif self.pad is False and need_pad_num !=0 and need_pad_num>self.rank:
            if len(batches):
                batches[-1].pop(-1)
            if len(batches[-1])==0:
                batches.pop(-1)

        assert sum(map(len, batches)) == self.num_left_samples

        if self.drop_last and len(batches) >= 1 and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        # 暂时弃用
        # self.num_consumed_samples_array = NumConsumedSamplesArray(buffer_size=os.environ.get(FASTNLP_DEQUE_SIZE, 30),
        #                                                          num_consumed_samples=self.num_consumed_samples)
        for batch in batches:
            self.num_consumed_samples += self.num_replicas * len(batch)
            yield list(map(int, batch))
        self.during_iter = False
        self.num_consumed_samples = 0
        self.old_batch_size = self.batch_size
        self.old_num_batch_per_bucket = self.num_batch_per_bucket
        self.old_num_replicas = self.num_replicas
        if self.epoch < 0:  # 防止用户没有修改epoch，导致每个epoch都一样了
            self.epoch -= 1

    def bucketerize(self, sorted_indices, batch_size, num_batch_per_bucket, seed):
        """
        将 indices 分桶

        :param sorted_indices: List[int]
        :param batch_size: int
        :param num_batch_per_bucket: int
        :param seed: int
        :return:  List[List[int]]
        """
        # 实际的 bucket 大小
        bucket_size = min(len(sorted_indices), batch_size * num_batch_per_bucket)
        rng = np.random.default_rng(abs(seed))
        num_buckets = (len(sorted_indices) + bucket_size - 1) // bucket_size
        batches = []
        batch_indices = []
        for i in range(num_buckets):
            bucket = sorted_indices[i * bucket_size:(i + 1) * bucket_size]
            rng.shuffle(bucket)  # bucket 内部 shuffle 一下
            _num_batches = len(bucket) // batch_size
            if _num_batches == 0:
                _batches = [bucket]
            else:
                _batches = np.array_split(bucket[:_num_batches*batch_size], _num_batches)
                if len(bucket) % batch_size != 0:
                    _batches.append(bucket[_num_batches*batch_size:])
            batch_indices.extend(list(range(len(batches), len(batches) + len(_batches))))
            batches.extend(_batches)
        last_batches = []
        # 最后一个batch 统一不参与shuffle，因为有的rank最后一个 batch 可能不足一个batch_size （不足的时候
        #  一定要放在末尾，所以就干脆所有的rank都不对最后一个batch进行shuffle）。
        if len(batches) >= 1:
            last_batches = [list(batches[-1])]
        batch_indices = list(batch_indices[:-1])
        rng = np.random.default_rng(abs(seed))  # 这里防止由于bucket长度不同，对随机数状态有影响
        rng.shuffle(batch_indices)  # 不同的 batch 也 shuffle ，当前这种可以保证每张卡上每个 batch 长度都接近的。
        batches = (np.array(batches, dtype=object)[batch_indices]).tolist()
        if last_batches:
            batches = batches + last_batches
        return batches

    def state_dict(self) -> Dict:
        if self.old_batch_size != self.batch_size or self.old_num_batch_per_bucket != self.num_batch_per_bucket:
            raise RuntimeError("BucketedBatchSampler does not support saving before last checkpoint states have been"
                               " consumed. ")
        states = {'seed': self.seed, 'epoch': self.epoch, 'num_consumed_samples': self.num_consumed_samples,
                  'sampler_type': self.__class__.__name__, 'length': len(self.dataset), 'shuffle': self.shuffle,
                  'batch_size': self.batch_size, 'num_batch_per_bucket': self.num_batch_per_bucket,
                  'num_replicas': self.num_replicas
                  }

        return states

    def load_state_dict(self, states: Dict):
        # 如果 self.during_iter 是 True，那么 num_consumed_samples 一定是 0；
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
        if self.shuffle != states['shuffle']:
            logger.info(f"The shuffle from the checkpoint is {states['shuffle']}, while set as {self.shuffle}, "
                        f"we use shuffle={states['shuffle']}")
        self.shuffle = states["shuffle"]
        self.old_batch_size = states['batch_size']
        self.old_num_batch_per_bucket = states['num_batch_per_bucket']
        self.old_num_replicas = states['num_replicas']

    def set_epoch(self, epoch):
        self.epoch = epoch

    @property
    def batch_idx_in_epoch(self):
        if self.drop_last:
            return len(self.dataset) // self.num_replicas // self.batch_size - self.num_left_samples // self.batch_size
        else:
            return (len(self.dataset) // self.num_replicas + self.batch_size - 1) // self.batch_size - \
                   (self.num_left_samples + self.batch_size - 1) // self.batch_size