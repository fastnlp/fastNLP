import array
import numpy as np
from typing import Union, List, Iterable, Dict

__all__ = [
    'MixSampler',
    'DopedSampler',
    'MixSequentialSampler',
    'PollingSampler'
]

from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    from torch.utils.data import SequentialSampler, Sampler
    import torch


class MixSampler:
    """
    所有 mix_sampler 的基类。

    :param dataset: 一个字典，每个元素都是一个实现了 __getitem__ 和 __len__ 的数据容器
    :param batch_size: ``dataset`` 的批次大小，所有 ``dataset`` 均采用该 ``batch_size`` 作为批次大小
    :param sampler: 实例化好的 ``sampler`` ，每个 ``dataset`` 对应一个 ``sampler`` 对象
    :param ds_ratio: ``ds_ratio`` 是控制 datasets 怎么组成一个混合大数据集的重要参数， 其取值为 ``[None, 'truncate_to_least', 'pad_to_most', List[float], Dict[str, float]]``:

        * ds_ratio 为 ``None``, datasets 数据集序列或字典不进行数据扩充处理；
        * ds_ratio 为 ``'truncate_to_least'``, datasets 数据集序列或字典会计算得到 datasets序列中 dataset 最断长度 ``mix_len``， 其他数据集会被切断
          到最短长度 ``mix_len``。这种切断不是物理上切断，``MixDataLoader`` 会根据 sampler 不同来采样数据集到指定的最短长度 ``mix_len``；
        * ds_ratio 为 ``'pad_to_most'``, datasets 数据集序列或字典会计算得到 datasets序列中 dataset 最大长度 ``max_len``, 其他其他数据集会扩充
          到最大长度 ``mix_len``。这种扩充不是物理上扩充， ``MixDataLoader`` 会根据 sampler 不同来重采样 dataset 到指定的最大长度 ``max_len``；
        * ds_ratio 为 ``Dict[str, float]`` 时， datasets 类型也必须为 ``Dict[str, DataSet]``, 其 key 一一对应。 ``ds_ratio`` 的 value 是任意大于 0 的浮点数，
          代表着 datasets 的 value 数据进行扩充或者缩减的倍数；

    :param drop_last: 当最后一个 batch 长度小于 ``batch_size`` 时是否丢弃
    :param rank: 分布式训练中当前进程的 ``global_rank``
    :param world_size: 分布式训练中进程的总数 **world_size**
    """

    def __init__(self, dataset: Dict, batch_size: int = None,
                 sampler: Union[Dict[str, "Sampler"], None, str] = None,
                 ds_ratio: Union[str, Dict[str, float]] = None,
                 drop_last: bool = False, rank: int = -1, word_size: int = -1) -> None:
        # sampler 为 dict，则判断是否与 datasets 的 key 相同
        if isinstance(sampler, Dict):
            for key in dataset.keys():
                if not sampler[key]:
                    raise ValueError(f"the key:{key} of datasets is not in sampler, where sampler is a dict!")

        if batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        if not isinstance(sampler, str) and (rank >= 0 or word_size >= 0):
            raise ValueError("if rank>=0 and word_size>=0, sampler must be str")

        if sampler is None and (word_size < 0 or rank < 0):
            self.sampler = {name: SequentialSampler(ds) for name, ds in dataset.items()}

        elif isinstance(sampler, Dict):
            self.sampler = sampler

        else:
            # 单卡多机情况下， sampler为None或者str且word_size>0, rank > 0
            if isinstance(sampler, str):
                if sampler not in ['seq', 'rand']:
                    raise ValueError(f"sampler is {sampler}, but seq or rand is required")
            self.sampler = sampler

        # 计算扩展后的大数据集长度total_len和扩展后的单个数据集长度sampler_len
        sampler_lens, total_lens, sampler_index = [], 0, []
        if isinstance(self.sampler, Dict):
            if ds_ratio is None:
                sampler_lens = [len(spl) for _, spl in self.sampler.items()]

            elif ds_ratio == 'pad_to_most':
                sampler_len = sum([1 for _ in self.sampler.keys()])
                sampler_lens = [max(len(spl) for _, spl in self.sampler.items())] * sampler_len

            elif ds_ratio == 'truncate_to_least':
                sampler_len = sum([1 for _ in self.sampler.keys()])
                sampler_lens = [min(len(spl) for _, spl in self.sampler.items())] * sampler_len

            elif isinstance(ds_ratio, Dict):
                if not all([item >= 0 for item in ds_ratio.values()]):
                    raise ValueError("batch_size should be a positive integer value, "
                                     "but got ds_ratio={}".format(ds_ratio))
                sampler_lens = [int(len(spl) * ds_ratio[name]) for name, spl in self.sampler.items()]
            else:
                raise ValueError(f"{ds_ratio} must be pad_to_least or truncate_to_least or None or List")
            total_lens = sum(sampler_lens)

        # sampler 为 str 时候，初始化下移到 iter 方法中
        if len(sampler_lens) > 0:
            sampler_index = [sampler_lens[0]]
            for idx in sampler_lens[1:]:
                temp = sampler_index[-1]
                sampler_index.append(temp + idx)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.ds_ratio = ds_ratio
        self.rank = rank
        self.word_size = word_size
        self.datasets = dataset
        self.num_samplers = sampler_index
        self.len_samplers = total_lens
        self.epoch = 0

    def __iter__(self):
        pass

    def __len__(self):
        pass

    def set_epoch(self, epoch: int) -> None:
        """
        配合ddp使用， 控制随机数种子

        :param epoch: 当前的轮次
        :return:
        """
        self.epoch = epoch


class InnerSampler:
    """
    提供多卡情况下使用的内部 sampler
    """
    def __init__(self, ds_ind_list: List) -> None:
        self.ds_ind_list = ds_ind_list

    def __iter__(self) -> int:
        for item in self.ds_ind_list:
            yield item

    def __len__(self) -> int:
        return len(self.ds_ind_list)


class DopedSampler(MixSampler):
    """
    定制给 :class:`~fastNLP.core.dataloaders.MixDataLoader` 的 ``BatchSampler``，其功能是将传入的 ``datasets`` 
    字典混合采样组成一个个 batch 返回。
    """
    def __init__(self, dataset: Dict, batch_size: int = None,
                 sampler: Union[Dict[str, "Sampler"], str] = None,
                 ds_ratio: Union[str, None, Dict[str, float]] = None,
                 drop_last: bool = False, rank: int = -1, word_size: int = -1) -> None:
        super(DopedSampler, self).__init__(dataset=dataset, batch_size=batch_size,
                                           sampler=sampler, ds_ratio=ds_ratio,
                                           drop_last=drop_last, rank=rank, word_size=word_size)

    def __iter__(self) -> List[int]:
        # sampler 为 str， 此时为单机多卡或者单机，可以实现 rand 随机化
        if isinstance(self.sampler, str):
            if self.sampler == 'seq':
                self.sampler = {}
                for name, per_ds in self.datasets.items():
                    if self.word_size >= 0 and self.rank >= 0:
                        self.sampler[name] = InnerSampler(list(range(len(per_ds)))[self.rank::self.word_size])
                    else:
                        self.sampler[name] = InnerSampler(list(range(len(per_ds))))
            elif self.sampler == 'rand':
                self.sampler = {}
                for name, per_ds in self.datasets.items():
                    g = torch.Generator()
                    g.manual_seed(self.epoch)
                    indices = torch.randperm(len(per_ds), generator=g).tolist()
                    if self.word_size >= 0 and self.rank >= 0:
                        self.sampler[name] = InnerSampler(indices[self.rank::self.word_size])
                    else:
                        self.sampler[name] = InnerSampler(indices)

            # 根据给定的ds_ratio计算真正需要处理数据集
            if isinstance(self.sampler, Dict):
                if self.ds_ratio is None:
                    sampler_lens = [len(spl) for _, spl in self.sampler.items()]

                elif self.ds_ratio == 'pad_to_most':
                    sampler_len = sum([1 for _ in self.sampler.keys()])
                    sampler_lens = [max(len(spl) for _, spl in self.sampler.items())] * sampler_len

                elif self.ds_ratio == 'truncate_to_least':
                    sampler_len = sum([1 for _ in self.sampler.keys()])
                    sampler_lens = [min(len(spl) for _, spl in self.sampler.items())] * sampler_len

                elif isinstance(self.ds_ratio, Dict):
                    if not all(item >= 0 for item in self.ds_ratio):
                        raise ValueError("batch_size should be a positive integer value, "
                                         "but got ds_ratio={}".format(self.ds_ratio))
                    sampler_lens = [int(len(spl) * self.ds_ratio[name]) for name, spl in self.sampler.items()]
                else:
                    raise ValueError(f"{self.ds_ratio} must be pad_to_least or truncate_to_least or None or List")
                total_lens = sum(sampler_lens)
            else:
                raise ValueError("datasets must be dict or list")
            # 初始化参数
            sampler_index = [sampler_lens[0]]
            for idx in sampler_lens[1:]:
                temp = sampler_index[-1]
                sampler_index.append(temp + idx)
            self.num_samplers = sampler_index
            self.len_samplers = total_lens
        # 每个 batch 的数据, 总的数据量 total_index , 每个数据集的 samplers
        batch_idx, samplers = [], []
        # 如果单机则用所有数据，否则采用多卡
        if self.rank < 0 or self.word_size < 0:
            # 根据 sampler 长度判断是否使用 unsigned int 或者 unsigned long
            if self.len_samplers > 42e8:
                total_index = array.array('L', list(range(self.len_samplers)))
            else:
                total_index = array.array('I', list(range(self.len_samplers)))
        else:
            if (self.len_samplers // self.word_size) > 42e8:
                # 整分给每个卡的数据
                self.len_samplers = self.len_samplers - self.len_samplers % self.word_size
                total_index = array.array('L', list(range(self.len_samplers))[self.rank::self.word_size])
            else:
                total_index = array.array('I', list(range(self.len_samplers))[self.rank::self.word_size])

        start_idx = 0

        # （特定数据集需要长度，特定数据集sampler, 特定数据集的基址， 特定sampler的下标）
        for idx, (name, spl) in enumerate(self.sampler.items()):
            end_idx = len(spl)
            samplers.append((iter(spl), name, start_idx))
            start_idx += end_idx
        # 根据sampler的类型取出每个数据集的sampler
        # sampler_base_index = [0] + [len(spl) for _, spl in self.sampler.items()][:-1]
        # samplers = [(iter(spl), name, sampler_base_index[idx])
        #             for idx, (name, spl) in enumerate(self.sampler.items())]
        # 生成随机数
        np.random.seed(self.epoch)
        np.random.shuffle(total_index)
        for idx in total_index:
            ds_index = np.searchsorted(self.num_samplers, idx, side='right')
            spl, name, base_index = samplers[ds_index]
            try:
                batch_idx.append(next(spl) + base_index)
            except StopIteration:
                # 重新初始化一个新的sampler，因为不可能为空，故一定不会出现stopIteration
                spl = iter(self.sampler[name])
                batch_idx.append(next(spl) + base_index)
                samplers[ds_index] = (spl, name, base_index)
            if len(batch_idx) == self.batch_size:
                yield batch_idx
                batch_idx = []

        if len(batch_idx) > 0 and not self.drop_last:
            yield batch_idx

    def __len__(self) -> int:
        # 多卡情况下
        if self.rank >= 0 and self.word_size >= 0:
            # 整分给每个卡的数据
            self.len_samplers = (self.len_samplers - self.len_samplers % self.word_size) / self.word_size
        if self.drop_last:
            return self.len_samplers // self.batch_size
        else:
            return (self.len_samplers + self.batch_size - 1) // self.batch_size


class MixSequentialSampler(MixSampler):
    """
    定制给 :class:`~fastNLP.core.dataloaders.MixDataLoader` 的 ``BatchSampler``，其功能是将传入的 ``datasets`` 按顺序采样并返回 index，
    只有上一个 dataset 处理结束后才会处理下一个。
    """

    def __init__(self, dataset: Dict, batch_size: int = None,
                 sampler: Union[List["Sampler"], Dict[str, "Sampler"], None, str] = None,
                 ds_ratio: Union[str, List[float], Dict[str, float]] = None,
                 drop_last: bool = False, rank: int = -1, word_size: int = -1) -> None:
        super(MixSequentialSampler, self).__init__(dataset=dataset, batch_size=batch_size,
                                                   sampler=sampler, ds_ratio=ds_ratio,
                                                   drop_last=drop_last, rank=rank, word_size=word_size)

    def __iter__(self) -> Iterable[List[int]]:
        """
        按照 ``dataset`` 的顺序采样，打包成一个 batch 后返回。

        :return:
        """
        # sampler为str， 此时为单机多卡或者单机，可以实现rand随机化
        if isinstance(self.sampler, str):
            if self.sampler == 'seq':
                self.sampler = {}
                for name, per_ds in self.datasets.items():
                    if self.word_size >= 0 and self.rank >= 0:
                        self.sampler[name] = InnerSampler(list(range(len(per_ds)))[self.rank::self.word_size])
                    else:
                        self.sampler[name] = InnerSampler(list(range(len(per_ds))))
            elif self.sampler == 'rand':

                self.sampler = {}
                for name, per_ds in self.datasets.items():
                    g = torch.Generator()
                    g.manual_seed(self.epoch)
                    indices = torch.randperm(len(per_ds), generator=g).tolist()
                    if self.word_size >= 0 and self.rank >= 0:
                        self.sampler[name] = InnerSampler(indices[self.rank::self.word_size])
                    else:
                        self.sampler[name] = InnerSampler(indices)

            # 根据给定的 ds_ratio 算真正需要处理数据集
            if isinstance(self.sampler, Dict):
                if self.ds_ratio is None:
                    sampler_lens = [len(spl) for _, spl in self.sampler.items()]

                elif self.ds_ratio == 'pad_to_most':
                    sampler_len = sum([1 for _ in self.sampler.keys()])
                    sampler_lens = [max(len(spl) for _, spl in self.sampler.items())] * sampler_len

                elif self.ds_ratio == 'truncate_to_least':
                    sampler_len = sum([1 for _ in self.sampler.keys()])
                    sampler_lens = [min(len(spl) for _, spl in self.sampler.items())] * sampler_len

                elif isinstance(self.ds_ratio, Dict):
                    if not all(item >= 0 for item in self.ds_ratio):
                        raise ValueError("batch_size should be a positive integer value, "
                                         "but got ds_ratio={}".format(self.ds_ratio))
                    sampler_lens = [int(len(spl) * self.ds_ratio[name]) for name, spl in self.sampler.items()]
                else:
                    raise ValueError(f"{self.ds_ratio} must be pad_to_least or truncate_to_least or None or List")
                total_lens = sum(sampler_lens)
            else:
                raise ValueError("datasets must be dict or list")
            # 初始化参数
            sampler_index = [sampler_lens[0]]
            for idx in sampler_lens[1:]:
                temp = sampler_index[-1]
                sampler_index.append(temp + idx)
            self.num_samplers = sampler_index
            self.len_samplers = total_lens

        batch_idx, total_index, samplers = [], list(range(self.len_samplers)), []
        start_idx = 0

        # （特定数据集需要长度，特定数据集sampler, 特定数据集的基址， 特定sampler的下标）
        for idx, (name, spl) in enumerate(self.sampler.items()):
            end_idx = len(spl)
            samplers.append((iter(spl), name, start_idx))
            start_idx += end_idx
        # if self.word_size > 0 and self.rank >= 0:
        #     sampler_base_index = [0] + [len(spl) * self.word_size for _, spl in self.sampler.items()][:-1]
        # else:
        #     sampler_base_index = [0] + [len(spl) for _, spl in self.sampler.items()][:-1]
        #
        # samplers = [(iter(spl), name, sampler_base_index[idx])
        #             for idx, (name, spl) in enumerate(self.sampler.items())]
        for idx in total_index:
            ds_index = np.searchsorted(self.num_samplers, idx, side='right')

            spl, name, base_index = samplers[ds_index]
            try:
                batch_idx.append(next(spl) + base_index)
            except StopIteration:
                # 重新初始化一个新的sampler，因为不可能为空，故一定不会出现stopIteration
                spl = iter(self.sampler[name])
                batch_idx.append(next(spl) + base_index)
                samplers[ds_index] = (spl, name, base_index)
            if len(batch_idx) == self.batch_size:
                yield batch_idx
                batch_idx = []
            # 当前数据集采样完，需要及时处理最后一个batch
            if self.num_samplers[ds_index] == (idx + 1):
                if len(batch_idx) > 0 and not self.drop_last:
                    yield batch_idx
                batch_idx = []

    def __len__(self) -> int:
        lens, index = 0, 0
        num_sampler = []
        for ds_len in self.num_samplers:
            num_sampler.append(ds_len - index)
            index = ds_len

        for ds_len in num_sampler:
            if self.drop_last:
                lens += ds_len // self.batch_size
            else:
                lens += (ds_len + self.batch_size - 1) // self.batch_size
        return lens


class PollingSampler(MixSampler):
    """
    定制给 :class:`~fastNLP.core.dataloaders.MixDataLoader` 的 ``BatchSampler``，其功能是将传入的 ``datasets`` 轮流采样并返回 index，
    处理结束上个 dataset 的一个 batch 后会处理下一个。
    """

    def __init__(self, dataset: Union[List, Dict], batch_size: int = 16,
                 sampler: Union[List["Sampler"], Dict[str, "Sampler"], str] = None,
                 drop_last: bool = False, ds_ratio="pad_to_most", rank: int = -1,
                 word_size: int = -1) -> None:
        super(PollingSampler, self).__init__(dataset=dataset, batch_size=batch_size,
                                             sampler=sampler, ds_ratio=ds_ratio,
                                             drop_last=drop_last, rank=rank, word_size=word_size)

    def __iter__(self) -> List[int]:
        # sampler为str， 此时为单机多卡或者单机，可以实现rand随机化
        if isinstance(self.sampler, str):
            if self.sampler == 'seq':
                self.sampler = {}
                for name, per_ds in self.datasets.items():
                    if self.word_size >= 0 and self.rank >= 0:
                        self.sampler[name] = InnerSampler(list(range(len(per_ds)))[self.rank::self.word_size])
                    else:
                        self.sampler[name] = InnerSampler(list(range(len(per_ds))))
            elif self.sampler == 'rand':

                self.sampler = {}
                for name, per_ds in self.datasets.items():
                    g = torch.Generator()
                    g.manual_seed(self.epoch)
                    indices = torch.randperm(len(per_ds), generator=g).tolist()
                    if self.word_size >= 0 and self.rank >= 0:
                        self.sampler[name] = InnerSampler(indices[self.rank::self.word_size])
                    else:
                        self.sampler[name] = InnerSampler(indices)

            # 根据给定的ds_ratio计算真正需要处理数据集
            if isinstance(self.sampler, Dict):
                if self.ds_ratio is None:
                    sampler_lens = [len(spl) for _, spl in self.sampler.items()]

                elif self.ds_ratio == 'pad_to_most':
                    sampler_len = sum([1 for _ in self.sampler.keys()])
                    sampler_lens = [max(len(spl) for _, spl in self.sampler.items())] * sampler_len

                elif self.ds_ratio == 'truncate_to_least':
                    sampler_len = sum([1 for _ in self.sampler.keys()])
                    sampler_lens = [min(len(spl) for _, spl in self.sampler.items())] * sampler_len

                elif isinstance(self.ds_ratio, Dict):
                    if not all(item >= 0 for item in self.ds_ratio):
                        raise ValueError("batch_size should be a positive integer value, "
                                         "but got ds_ratio={}".format(self.ds_ratio))
                    sampler_lens = [int(len(spl) * self.ds_ratio[name]) for name, spl in self.sampler.items()]
                else:
                    raise ValueError(f"{self.ds_ratio} must be pad_to_least or truncate_to_least or None or List")
                total_lens = sum(sampler_lens)
            else:
                raise ValueError("datasets must be dict or list")
            # 初始化参数
            sampler_index = [sampler_lens[0]]
            for idx in sampler_lens[1:]:
                temp = sampler_index[-1]
                sampler_index.append(temp + idx)
            self.num_samplers = sampler_index
            self.len_samplers = total_lens

        start_idx, samplers, true_start_idx, true_end_idx = 0, [], 0, 0

        # （特定数据集需要长度，特定数据集sampler, 特定数据集的基址， 特定sampler的下标）
        for idx, (name, spl) in enumerate(self.sampler.items()):
            end_idx = len(spl)
            true_end_idx = self.num_samplers[idx]
            samplers.append((iter(range(true_start_idx, true_end_idx)), iter(spl), start_idx, name))
            start_idx += end_idx
            true_start_idx = true_end_idx

        while True:
            # 退出循环
            if len(samplers) == 0:
                break
            batch_idx, flag = [], False
            ds_total_iter, ds_sampler, ds_base_idx, sampler_idx = samplers.pop(0)
            for _ in range(self.batch_size):
                try:
                    # 取出数据
                    next(ds_total_iter)
                    # 取出真正数据， 若取完则重新初始化一个
                    try:
                        batch_idx.append(next(ds_sampler) + ds_base_idx)
                    except StopIteration:
                        ds_sampler = iter(self.sampler[sampler_idx])
                        batch_idx.append(next(ds_sampler) + ds_base_idx)
                except StopIteration:
                    # 当前ds所有的数据集采样完毕，将其清除队列
                    flag = True
            # 判断是否真正解决某个数据集的采样
            if flag is False:
                samplers.append((ds_total_iter, ds_sampler, ds_base_idx, sampler_idx))
            if len(batch_idx) == self.batch_size:
                yield batch_idx
            elif len(batch_idx) > 0 and not self.drop_last:
                yield batch_idx

    def __len__(self) -> int:
        lens, index = 0, 0
        num_sampler = []
        for ds_len in self.num_samplers:
            num_sampler.append(ds_len - index)
            index = ds_len

        for ds_len in num_sampler:
            if self.drop_last:
                lens += ds_len // self.batch_size
            else:
                lens += (ds_len + self.batch_size - 1) // self.batch_size
        return lens


if __name__ == '__main__':
    from fastNLP.core.dataset import DataSet
    ds = DataSet({'x': ["x1a", "1ws2", "xa qa", "ax wq", "iu, lk"] * 101, 'y': [1, 0, 1, 0, 1] * 101})
    ds1 = DataSet({'x': ["x12a", "1wzs2", "xa xqa", "aax wq", "iau, lk"] * 101, 'y': ['1', '0', '1', '0', '1'] * 101})
    sampler = DopedSampler(dataset=[ds, ds1], batch_size=6, rank=0, word_size=-2, sampler='seq')
    seqSpl = MixSequentialSampler(dataset=[ds, ds1], batch_size=6, rank=0, word_size=2, sampler='seq', drop_last=True)
    polSpl = PollingSampler(dataset=[ds, ds1], batch_size=6, rank=1, word_size=2, sampler='seq', drop_last=False)
    for idx, batch in enumerate(polSpl):
        print(idx, batch)
    # print(len(seqSpl))
