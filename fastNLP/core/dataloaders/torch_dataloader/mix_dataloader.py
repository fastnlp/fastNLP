__all__ = [
    'MixDataLoader'
]

from typing import Optional, Callable, List, Union, Tuple, Dict, Sequence, Mapping

import numpy as np
from pkg_resources import parse_version

from fastNLP.core.dataset import DataSet, Instance
from fastNLP.core.samplers import PollingSampler, MixSequentialSampler, DopedSampler
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.core.collators import Collator

if _NEED_IMPORT_TORCH:
    from torch import __version__ as torchversion
    from torch.utils.data import DataLoader, Sampler
else:
    from fastNLP.core.utils.dummy_class import DummyClass as DataLoader


class _MixDataset:
    """
    将所有数据集当成一个混合大数据集来对待， 在 __getitem__() 能根据输入的 idx 来判断属于哪个小数据并返回其 ds_index

    """

    def __init__(self, datasets: list = None) -> None:
        """
        :param datasets: 实现了 __getitem__() 和 __len__() 的对象的序列
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
        根据index索引获取数据， 能够跟 idx 的范围定位属于哪个小数据并返回

        :param idx: 整数类型的index或者列表
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

    def __init__(self, collate_fns: Union[List[Callable], Callable]) -> None:

        if isinstance(collate_fns, Sequence):
            self.collate_fns = lambda idx, lst: collate_fns[idx](lst)
        elif callable(collate_fns):
            self.collate_fns = lambda idx, lst: collate_fns(lst)
        else:
            self.collate_fns = lambda idx, lst: lst

    def __call__(self, ins_list: List) -> Dict:
        """
        调用一次该方法，我们将ins_list视为同一个数据集采样出来的，故ds_index只能为一种

        :param ins_list:
        :return:
        """
        _ins_list, _ds_index = [], 0
        for ins, _ds_index in ins_list:
            _ins_list.append(ins)
        _ins_list = self.collate_fns(_ds_index, _ins_list)
        return _ins_list


class MixDataLoader(DataLoader):
    """
    针对一下四种情况提供的 ``MixDataLoader``， 目前只支持 ``torch`` 框架的版本， 其中 mode 的取值范围为 ``['sequential', 'mix', 'polling', "Sampler"]``:

        * 当 mode 为 ``'sequential'`` 时，``MixDataLoader``  将 datasets 的序列或者字典视为一个混合大数据集， 按照 datasets 数据集序列或者字典的顺序一个
        接一个的 sample 完所有数据。
        * 当 mode 为 ``'mix'`` 时， ``MixDataLoader``  将 datasets 的序列或者字典视为一个混合大数据集， 然后根据用户输入的 idx 序列随机sample
        混合数据集 datasets 的数据组成一个 batch 序列返回。
        * 当 mode 为 ``'polling'`` 时， ``MixDataLoader`` 按照 datasets 数据集的顺序， 先从第一个数据集采样一个 batch 的数据返回，
        再从第二数据集采样一个 batch 数据返回， 直至最后一个数据集采样一个 batch 数据返回后再从第一个数据采样第二个 batch 数据返回，直至所有的数据集都被轮询的采样完。
        * 当 mode 为 ``"Sampler"`` 时， 该 Sampler 是实现 __iter__() 的实例化对象， 其功能是每次 iter 时返回一个 batch 序列， 其类型为 List[int];
        且 Sampler 必须将输入的 datasets 视为一个混合大数据集， 其 index 范围为 ``0<idx<len(datasets[0])+...+len(datasets[x])``, 然后参数
        sampler, drop_last, ds_ratio 均无效。

    """

    def __init__(self, datasets: Dict = None, mode: Union[str, "Sampler"] = 'sequential',
                 collate_fn: Union[str, Callable, Dict[str, Callable]] = 'auto',
                 sampler: Union[Dict[str, "Sampler"], str, None] = None,
                 num_workers: int = 0, batch_size: int = 16, drop_last=False,
                 ds_ratio: Union[None, str, Dict[str, float]] = None,
                 pin_memory: bool = False) -> None:
        """

        :param datasets: 实现了 __getitem__() 和 __len__() 对象的序列或者字典。
        :param mode: mode 控制 ``MixDataLoader`` 运行模式。 mode 的取值范围为 ``['sequential', 'mix', 'polling', "Sampler"]``:

            * 当 mode 为 ``'sequential'`` 时，``MixDataLoader``  将 datasets 的序列或者字典视为一个混合大数据集， 按照 datasets 数据集序列或者字典的顺序一个
            接一个的 sample 完所有数据。
            * 当 mode 为 ``'mix'`` 时， ``MixDataLoader``  将 datasets 的序列或者字典视为一个混合大数据集， 然后根据用户输入的 idx 序列随机sample
            混合数据集 datasets 的数据组成一个 batch 序列返回。
            * 当 mode 为 ``'polling'`` 时， ``MixDataLoader`` 按照 datasets 数据集的顺序， 先从第一个数据集采样一个 batch 的数据返回，
            再从第二数据集采样一个 batch 数据返回， 直至最后一个数据集采样一个 batch 数据返回后再从第一个数据采样第二个 batch 数据返回，直至所有的数据集都被轮询的采样完。
            * 当 mode 为 ``"Sampler"`` 时， 该 Sampler 是实现 __iter__() 的实例化对象， 其功能是每次 iter 时返回一个 batch 序列， 其类型为 List[int];
            且 Sampler 必须将输入的 datasets 视为一个混合大数据集， 其 index 范围为 ``0<idx<len(datasets[0])+...+len(datasets[x])``, 然后参数
            sampler, drop_last, ds_ratio 均无效。

        :param collate_fn: 用于从 dataset 取到的一个 batch 数据进行打包处理的 Callable 函数。 其取值可以为 ``['auto', Callable, List[Callable], Dict[str, Callable]]``:

            * collate_fn 为 ``'auto'`` 时, ``MixDataLoader``  datasets 序列或者dict 初始化一个 :class: `~fastNLP.core.collators.Collator`  作为其默认值，
            需要注意的是只有当 datasets 包含的所以 dataset 的数据都为 ``List`` 或者 ``Dict`` 类型时才能使用。否则只能用户自己定义 collate_fn .
            * collate_fn 为  ``Callable`` 时， 该 collate_fn 会被 datasets 序列或者dict 的所有数据所共享。该 Callable 函数应当接受一个 batch 参数作为输入，
            batch 是一个 List 对象且 List 中的每一条数据都是 dataset 的一条数据；该 Callable 函数还应当返回一个对象。
            * collate_fn 为 ``Dict[str, Callable]`` 时， datasets 的 key 必须和 callable_fn 的 key 一致。 ``MixDataLoader`` 会将 ``collate_fn[key]``
            用到 ``datasets[key]`` 的数据集上。 ``collate_fn[key]`` 是一个 Callable 对象。


        :param sampler: 实现了 __len__() 和 __iter__() 的实例化对象，其 __iter__() 方法每次都会返回 dataset 的一个下标 index ，其取值范围为
        ``[None, str, Dict[str, "Sampler"]]``:

            * sampler 为 ``None`` 时， ``MixDataLoader`` 默认初始化 ``torch`` 的 ``SequentialSampler`` 作为默认值。其功能时顺序返回 dataset 的下标。
            * sampler 为 ``str`` 时， sampler 选择范围为 ``[rand, seq]``。当 sampler 为 ``rand`` 时，``MixDataLoader`` 默认初始化 ``torch`` 的  ``RandomSampler``
            作为默认值， 其功能时随机采样 dataset 的下标并返回。 当 sampler 为 ``seq`` 时， ``MixDataLoader`` 默认初始化 ``torch`` 的 ``SequentialSampler`` 作为默认值。其功能时顺序返回 dataset 的下标。
            * sampler 为 ``Dict[str, "Sampler"]`` 时， ``Sampler`` 为用户定义的实现了 __len__() 和 __iter__() 的实例化对象。 其每次 iter 必须返回一个 int 下标。
            Dict 的 str 必须和 datasets 的 key 一致。 也即是 ``Dict[str, Sampler] `` 为 datasets 字典的每个 dataset 初始化勒一个 Sampler。

        :param num_workers: 当 ``num_workers > 0`` 时, ``MixDataLoader`` 会开启 num_workers 个子进程来处理数据， 可以加快数据处理速度，但同时
        也消耗大量内存。 当 ``num_workers=0`` 时， 不开启子进程。 默认为 ``0``。
        :param batch_size: 批次大小，默认为 ``16`` 且当 batch_sampler 为 None 有效。 且 datasets 上所有 dataset 的 batch_size 一致。
        :param drop_last: 当 ``drop_last=True`` 时，``MixDataLoader`` 会扔掉 datasets 中 每个 dataset 最后一个长度小于 ``batch_size`` 的 batch 数据;
        若 ``drop_last=False`` , 则会返回该 batch 数据。 默认为 ``False`` 。
        :param ds_ratio: ``ds_ratio`` 是控制 datasets 怎么组成一个混合大数据集的重要参数， 其取值为 ``[None, 'truncate_to_least', 'pad_to_most', List[float], Dict[str, float]]``:

            * ds_ratio 为 ``None``, datasets 数据集序列或字典不进行数据扩充处理。
            * ds_ratio 为 ``'truncate_to_least'``, datasets 数据集序列或字典会计算得到 datasets序列中 dataset 最断长度 ``mix_len``， 其他数据集会被切断
            到最短长度``mix_len``。这种切断不是物理上切断，``MixDataLoader`` 会根据 sampler 不同来采样数据集到指定的最短长度``mix_len``。
            * ds_ratio 为 ``'pad_to_most'``, datasets 数据集序列或字典会计算得到 datasets序列中 dataset 最大长度 ``max_len``, 其他其他数据集会扩充
            到最大长度``mix_len``。这种扩充不是物理上扩充， ``MixDataLoader`` 会根据 sampler 不同来重采样 dataset 到指定的最大长度``max_len``。
            * ds_ratio 为 ``Dict[str, float]`` 时， datasets 类型也必须为 ``Dict[str, DataSet]``, 其 key 一一对应。 ds_ratio 的 value 是任意大于 0 的浮点数，
            代表着 datasets 的 value 数据进行扩充或者缩减的倍数。
        """
        # sampler 为 dict，则判断是否与 datasets 的 key 相同
        if isinstance(sampler, Dict):
            for key in datasets.keys():
                if not sampler[key]:
                    raise ValueError(f"the key:{key} of datasets is not in sampler, where sampler is a dict!")
        # collate_fn 为 dict，则判断是否与 datasets 的 key 相同
        if isinstance(collate_fn, Dict):
            if mode == 'mix':
                raise ValueError(f"mode: {mode} do not support collate_fn is Dict, please use callate_fn=Callable or 'auto'")
            for key in datasets.keys():
                if not collate_fn[key]:
                    raise ValueError(f"the key:{key} of datasets is not in collate_fn, where collate_fn is a dict!")

        if isinstance(collate_fn, str) and collate_fn == 'auto':
            date_type = None
            for idx, ds in enumerate(datasets.values()):
                if idx == 0:
                    date_type = type(ds[0])
                if type(ds[0]) != date_type or not (isinstance(ds[0], List) or isinstance(ds[0], Mapping)):
                    raise ValueError(f"when you use callate_fn={collate_fn}, all dataset must be list or dict。"
                                     f"But dataset {idx - 1} data type is {date_type}, dataset {idx} data type is {type(ds[0])}")

            collate_fn = Collator(backend='torch')

        # Dict 类型的 collate_fn 转化为 List，以便于 _MixCollateFn 里面根据 idx 定位 dataset
        if isinstance(collate_fn, Dict):
            collate_fn = [fn for _, fn in collate_fn.items()]

        dataset = [ds for _, ds in datasets.items()]

        # 对 collate_fn 进行包裹， 统一处理 collate_fn 不同情况下使用的问题
        collate_fn = _MixCollateFn(collate_fn)

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

        if parse_version(torchversion) >= parse_version('1.7'):
            super(MixDataLoader, self).__init__(
                _MixDataset(datasets=dataset), batch_size=1, shuffle=False, sampler=None,
                batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn,
                pin_memory=pin_memory, drop_last=False, timeout=0,
                worker_init_fn=None, multiprocessing_context=None, generator=None,
                prefetch_factor=2, persistent_workers=False
            )
        else:
            super(MixDataLoader, self).__init__(
                _MixDataset(datasets=dataset), batch_size=1, shuffle=False, sampler=None,
                batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn,
                pin_memory=pin_memory, drop_last=False, timeout=0,
                worker_init_fn=None, multiprocessing_context=None, generator=None,
            )

    def __iter__(self):
        return super().__iter__()
