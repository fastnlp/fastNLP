__all__ = [
    'MixDataLoader'
]

from typing import Optional, Callable, List, Union, Tuple, Dict, Sequence

import numpy as np

from fastNLP.core.dataset import DataSet, Instance
from fastNLP.core.samplers import PollingSampler, MixSequentialSampler, DopedSampler
from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    from torch.utils.data import DataLoader, Sampler
else:
    from fastNLP.core.utils.dummy_class import DummyClass as DataLoader


class _MixDataset:
    """
    将所有数据集当成一个混合大数据集来对待，实现的__getitem__能区别每个数据idx

    """
    def __init__(self, datasets: list = None) -> None:
        """

        :param datasets: 数据集的列表
        """
        self.datasets = datasets
        # 记录每个数据集的长度索引， 以便根据idx定位数据集的位置
        self.lens = []
        index = 0
        for item in self.datasets:
            index += len(item)
            self.lens.append(index)

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Tuple[Instance, int], Tuple[DataSet, int]]:
        """

        :param idx:
        :return:
        """
        if isinstance(idx, int):
            if idx >= self.lens[-1]:
                raise ValueError(f"idx: {idx} out of range")
            # 找到其属于哪个数据集，返回下标
            ds_index = np.searchsorted(self.lens, idx, side='right')
            if ds_index > 0:
                idx -= self.lens[ds_index - 1]
            return self.datasets[ds_index][idx], ds_index
        elif isinstance(idx, list):
            # 一般一个list列表只能是属于一种数据的，否则会报错
            dataset = DataSet()
            ds_index = 0
            for i in idx:
                assert isinstance(i, int), "Only int index allowed."
                instance, ds_index = self[i]
                dataset.append(instance)
            return dataset, ds_index
        else:
            raise KeyError("Unrecognized type {} for idx in __getitem__ method".format(type(idx)))

    def __len__(self) -> int:
        return self.lens[-1]


class _MixCollateFn:
    """
    存在多个auto_collate和多个collate_fn时候，对一个批次数据集应用哪个auto_collate和collate_fn的问题

    """
    def __init__(self, collate_fns: Optional[Union[List[Callable], Callable]] = None,
                 auto_collators: Optional[List[Callable]] = None) -> None:
        if isinstance(collate_fns, Sequence):
            self.collate_fns = lambda idx, lst: collate_fns[idx](lst)
        elif callable(collate_fns):
            self.collate_fns = lambda idx, lst: collate_fns(lst)
        else:
            self.collate_fns = lambda idx, lst: lst

        self.collate_fns = collate_fns
        self.auto_collators = auto_collators

    def __call__(self, ins_list: List) -> Dict:
        """
        调用一次该方法，我们将ins_list视为同一个数据集采样出来的，故ds_index只能为一种
        :param ins_list:
        :return:
        """
        _ins_list, _ds_index = [], 0
        for ins, _ds_index in ins_list:
            _ins_list.append(ins)
        # auto_collate先处理
        if self.auto_collators is not None:
            _ins_list = self.auto_collators[_ds_index](_ins_list)
        _ins_list = self.collate_fns(_ds_index, _ins_list)
        return _ins_list


class MixDataLoader(DataLoader):
    """
    针对一下三种情况提供的MixDataLoader:
        1. 给定datasets集合或者列表，顺序采样datasets，处理采样完首个dataset后取出第二个dataset，重复上面过程直至datasets取完。
        2. 给定datasets集合或者列表，随机采样这个datasets的任意一个数据集组合成一个混合的batch返回给用户，直至datasets所有数据集采样完。
        3. 给定datasets集合或者列表，轮流采样datasets：即是循环遍历datasets，每取出一个dataset采样一个batch的数据，然后取出下一个dataset
           采样一个batch数据，重复上述过程直至某个dataset采样结束或者所有dataset采样结束。
    """
    def __init__(self, datasets: Union[List, Dict] = None, mode: Union[str, "Sampler"] = 'sequential',
                 collate_fn: Union[List[Callable], Callable, Dict[str, Callable]] = None,
                 sampler: Union[List["Sampler"], Dict[str, "Sampler"]] = None,
                 num_workers: int = 0, batch_size: int = 16, drop_last=False,
                 ds_ratio: Union[str, List[float], None, Dict[str, float]] = None,
                 pin_memory: bool = True) -> None:
        """

        :param datasets: dataset的列表
        :param mode: mode包括四种类型，前三种分别为"sequential", "mix", "polling"分别代表上述三种情况，
                    当mode为Sampler时为用户定制，此时sampler，ds_ratio，batch_size，drop_last失效，此时Sampler应该是一个可迭代
                    对象，每次迭代返回的是List[int]
        :param collate_fn: 对取得到的数据进行打包的callable函数，
                         当其为callable类型时候，所有数据集采样的数据都会经过这个函数；
                         当其为List[Callable]类型时，datasets也应该为List；会根据每个数据集__getitem__返回的idx判断当前数据对应的Callable函数，
                         其对应关系与datasets位置匹配；
                         当其为Dict[str, Callable]类型时， datasets也是Dict类型且一一对应。
        :param sampler: sampler是datasets每个数据集内部采样的实例化sampler对象
                        sampler为None时候，datasets包含的每个dataset都会初始化一个sequentialSampler用于采样;
                        sampler为List[Sampler]，则datasets也为List，且一一对应
                        sampler为Dict[str, Sampler], datasets也是Dict类型且一一对应。
        :param num_workers: 进程的数量，当num_workers=0时不开启多进程
        :param batch_size: 批次大小, datasets的所有数据集batch_size一致
        :param drop_last: 是否去掉最后一个不符合batch_size的数据
        :param ds_ratio: 当ds_ratio为None，原有数据集不进行扩充
                        当ds_ratio为'truncate_to_least'时，以datasets的最短数据集为基准，将其他数据集截断到一样长度
                        当ds_ratio为'pad_to_most'时，以datasets的最长数据集为基准，将最短数据集重采样到最长数据集长度一致为止
                        当ds_ratio为List[float]时，datasets也为List，ds_ratio的每一个参数都是datasets每个数据集应该采样的倍数，
                        其大于0，可以超过1，将数据集重采样翻倍即可
                        当ds_ratio为Dict[str, float]时，datasets也为Dict，参数相互对应。
        """
        # 如果dataset为Dict，则其他参数如collate_fn必须为Dict或者Callable,
        if not isinstance(datasets, Dict) and (isinstance(collate_fn, Callable) or isinstance(collate_fn, Dict)) and \
                isinstance(sampler, Dict):
            raise ValueError(f"")

        if isinstance(collate_fn, list):
            if len(collate_fn) != len(datasets):
                raise ValueError("the length of collate_fn != datasets!!")

        if isinstance(sampler, list):
            if len(sampler) != len(datasets):
                raise ValueError("the length of sampler != datasets!!")

        # Dict类型转化为List，以便于_MixCollateFn处理
        if isinstance(collate_fn, Dict):
            collate_fn = [fn for _, fn in collate_fn.items()]

        # 由于datasets可能是FastNLP类型的dataset或者是交杂的, 故需要检测
        if isinstance(datasets, Dict):
            dataset = [ds for _, ds in datasets.items()]
        else:
            dataset = datasets
        auto_collators = []
        for per_ds in dataset:
            if isinstance(per_ds, DataSet):
                auto_collators.append(per_ds.get_collator())
            else:
                # 如果没有对应的collator就设置一个不做任何操作的collator
                auto_collators.append(lambda x: x)

        # List类型的collate_fn只有两种情况，需要对其进行包裹
        collate_fn = _MixCollateFn(collate_fn, auto_collators)
        if mode == 'sequential':
            batch_sampler = MixSequentialSampler(datasets, batch_size=batch_size, sampler=sampler,
                                                 drop_last=drop_last, ds_ratio=ds_ratio)
        elif mode == 'polling':
            batch_sampler = PollingSampler(datasets, batch_size=batch_size, sampler=sampler,
                                           drop_last=drop_last, ds_ratio=ds_ratio)
        elif mode == 'mix':
            batch_sampler = DopedSampler(datasets, batch_size=batch_size, sampler=sampler,
                                         drop_last=drop_last, ds_ratio=ds_ratio)
        elif isinstance(mode, Sampler):
            batch_sampler = mode
        else:
            raise ValueError(f"{mode} must be sequential, polling, mix or batch_sampler")

        super(MixDataLoader, self).__init__(
            _MixDataset(datasets=dataset), batch_size=1, shuffle=False, sampler=None,
            batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=False, timeout=0,
            worker_init_fn=None, multiprocessing_context=None, generator=None,
            prefetch_factor=2, persistent_workers=False
        )

    def __iter__(self):
        return super().__iter__()
