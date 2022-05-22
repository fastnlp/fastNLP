__all__ = [
    'TorchDataLoader',
    'prepare_torch_dataloader'
]

from typing import Optional, Callable, Sequence, Union, Tuple, Dict, Mapping, List
from copy import deepcopy

from fastNLP.core.dataset import DataSet
from fastNLP.core.collators import Collator
from fastNLP.core.dataloaders.utils import indice_collate_wrapper
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleSampler, UnrepeatedSampler, RandomSampler
from ..utils import _match_param

if _NEED_IMPORT_TORCH:
    from torch.utils.data import DataLoader, Sampler
else:
    from fastNLP.core.utils.dummy_class import DummyClass as DataLoader


class _FDataSet:
    """
    提供给 ``TorchDataLoader`` 使用的 warp 类，其功能是对 dataset 进行封装，wrap 修改 dataset 的 __getitem__ 函数，增加返回
    数据的下标 idx 。

    ..note::

        需要注意的是传入 ``__init__`` 的 dataset 需要实现 __getattribute__ 方法才能在 _FDataset 实例化对象中调用 dataset 的方法

    """

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __getitem__(self, item: Union[int, list]) -> Tuple:
        return (item, self.dataset[item])

    def __getattr__(self, item):
        try:
            return self.dataset.__getattribute__(item)
        except AttributeError as e:
            raise e

    def __len__(self) -> int:
        return len(self.dataset)


class TorchDataLoader(DataLoader):
    """
    提供给 ``torch`` 框架使用的 ``DataLoader`` 函数，``TorchDataLoader`` 提供了 ``Collator`` 来自动检测 dataset 的每个 field 是否可 pad，
    若是可 pad 的 field 则自动 pad 到相同长度，否则只会将相同 field 的数据收集组成一个 batch 返回。
    具体详见 :class:`~fastNLP.core.collators.Collator`；用户通过 callte_fn 来控制是否使用该功能， collate_fn 只能为 ``['auto', None, Callable]``三种取值。

        * callate_fn 为 ``'auto'`` 时，``TorchDataLoader`` 使用 :class:`~fastNLP.core.collators.Collator` 作为 collate_fn 的取值。
        此时可以配套使用 ``TorchDataLoader`` 的 ``set_pad`` 和 ``set_ignore`` 方法来设置 pad_val 或 忽略某个 field 的检测。
        * callate_fn 为 ``None`` 时， ``TorchDataLoadr`` 默认使用 torch DataLoader 自带的 collate_fn
        * collate_fn 为 ``Callable`` 时， 该 Callable 函数应当接受一个 batch 参数作为输入， batch 是一个 List 对象且 List 中的每一条数据都是
         dataset 的一条数据；该 Callable 函数还应当返回一个对象。

    """

    def __init__(self, dataset, batch_size: int = 16,
                 shuffle: bool = False, sampler: Union["Sampler[int]", ReproducibleSampler, UnrepeatedSampler] = None,
                 batch_sampler: Union["Sampler[Sequence[int]]", ReproducibleBatchSampler] = None,
                 num_workers: int = 0, collate_fn: Union[Callable, str, None] = 'auto',
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[Callable] = None,
                 multiprocessing_context=None, generator=None, prefetch_factor: int = 2,
                 persistent_workers: bool = False, **kwargs) -> None:
        """

        :param dataset: 实现了 __getitem__() 和 __len__() 的对象。
        :param batch_size: 批次大小，默认为 ``16`` 且当 batch_sampler 为 None 有效。
        :param shuffle: 是否打乱数据集， 默认为 ``False``。
        :param sampler: 实现了 __len__() 和 __iter__() 的实例化对象，其 __iter__() 方法每次都会返回 dataset 的一个下标 index ，
        默认为None， 当其不为 None 时， shuffle 参数无效。
        :param batch_sampler: 实现了 __len__() 和 __iter__() 的实例化对象，，其__iter__() 方法每次都会返回一个 List 对象， List中的值为
        dataset 的下标 index ；默认为 None，当其不为 None 时，bacth_size, sampler, shuffle 参数均失效。
        :param num_workers: 当 ``num_workers > 0`` 时, ``TorchDataLoader`` 会开启 num_workers 个子进程来处理数据， 可以加快
        数据处理速度，但同时也消耗大量内存。 当 ``num_workers=0`` 时， 不开启子进程。 默认为 ``0``。
        :param collate_fn: 用于从 dataset 取到的一个 batch 数据进行打包处理的 Callable 函数，其值应该为以下三个: ``[None, "auto", Callable]``.

            *  callate_fn 为 ``None`` 时，需要注意的是此时传进来的 datset 类型不能为 :class:`~fastNLP.core.dataset.DataSet` , 当 collate_fn 为 ``None`` 时，
             ``TorchDataLoader`` 调用默认的 torch 框架的 ``DataLoader`` 自带的 ``default_collate_fn`` 作为 callate_fn 的默认值， 其无法处理
             :class:`~fastNLP.core.dataset.DataSet` 的dataset对象。
            * callate_fn 为 ``'auto'`` 时，``TorchDataLoader`` 使用 :class:`~fastNLP.core.collators.Collator` 作为 collate_fn 的默认值。
            此时可以配套使用 ``TorchDataLoader`` 的 ``set_pad`` 和 ``set_ignore`` 方法来设置 pad_val 或 忽略某个 field 的检测。
            * `collate_fn 为 ``Callable`` 时， 该 Callable 函数应当接受一个 batch 参数作为输入， batch 是一个 List 对象且 List 中的每一条数据都是
            dataset 的一条数据；该 Callable 函数还应当返回一个对象。

        :param pin_memory: 如果其为 ``True``, 那么 ``TorchDataLoader`` 会在返回数据张量之前将其 copy 到 cud a的 pin memory 中。
        :param drop_last: 当 ``drop_last=True`` 时，``TorchDataLoader`` 会扔掉最后一个长度小于 ``batch_size`` 的 batch 数据;
        若 ``drop_last=False`` , 则会返回该 batch 数据。 默认为 ``False`` 。
        :param timeout: 子进程的输出队列获取数据的超时值
        :param worker_init_fn: init 函数，如果不设置为 None ,则将会在每个子进程初始化时调用该函数。
        :param multiprocessing_context: 多进程的上下文环境
        :param generator: 如果其不为 ``None``, 将会使用 RandomSampler 去生成随机的 index 且会为每个子进程生成一个``base_seed``
        :param prefetch_factor: 每个 worker 提前装载的 samples 数量。``2``意味着在所有的进程中会有 2*num_workers 的数据被预取。默认值为 ``2`` .
        :param persistent_workers: 如果其为 ``True``, ``TorchDataLoader`` 在迭代完一次 dataset 后不会关闭所有进程。默认为 ``False``

        """
        if isinstance(dataset, DataSet) and collate_fn is None:
            raise ValueError("When use FastNLP DataSet, collate_fn must be not None")

        if not isinstance(dataset, _FDataSet):
            dataset = _FDataSet(dataset)

        if batch_sampler is not None:
            batch_size = 1
            shuffle = False
            sampler = None
        elif sampler is None:
            sampler = RandomSampler(dataset, shuffle=shuffle)
            shuffle = False

        if isinstance(collate_fn, str):
            if collate_fn == 'auto':
                if isinstance(dataset.dataset, DataSet):  # 使用了 fastnlp dataset
                    collate_fn = deepcopy(dataset.dataset.collator)
                    collate_fn.set_backend(backend="torch")
                else:
                    collate_fn = Collator(backend="torch")
            else:
                raise ValueError(f"collate_fn: {collate_fn} must be 'auto'")

        dl_kwargs = _match_param(TorchDataLoader.__init__, DataLoader.__init__, fn_name=DataLoader.__name__)
        if dl_kwargs is None:
            super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                             batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn,
                             pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                             multiprocessing_context=multiprocessing_context, generator=generator,
                             prefetch_factor=prefetch_factor,
                             persistent_workers=persistent_workers)
        else:
            super().__init__(**dl_kwargs)

        self.cur_batch_indices = None

    def __iter__(self):
        self.collate_fn = indice_collate_wrapper(self.collate_fn)
        for indices, data in super().__iter__():
            self.cur_batch_indices = indices
            yield data

    def set_pad(self, field_name: Union[str, tuple], pad_val: Union[int, float, None] = 0, dtype=None, backend=None,
                pad_fn: Callable = None) -> Collator:
        """
        如果需要对某个 field 的内容进行特殊的调整，请使用这个函数。

        :param field_name: 需要调整的 field 的名称。如果 Dataset 的 __getitem__ 方法返回的是 dict 类型的，则可以直接使用对应的
            field 的 key 来表示，如果是 nested 的 dict，可以使用元组表示多层次的 key，例如 {'a': {'b': 1}} 中的使用 ('a', 'b');
            如果 __getitem__ 返回的是 Sequence 类型的，则可以使用 '_0', '_1' 表示序列中第 0 或 1 个元素。如果该 field 在数据中没
            有找到，则报错；如果 __getitem__ 返回的是就是整体内容，请使用 "_single" 。
        :param pad_val: 这个 field 的默认 pad 值。如果设置为 None，则表示该 field 不需要 pad , fastNLP 默认只会对可以 pad 的
            field 进行 pad，所以如果对应 field 本身就不是可以 pad 的形式，可以不需要主动设置为 None 。如果 backend 为 None ，该值
            无意义。
        :param dtype: 对于需要 pad 的 field ，该 field 的数据 dtype 应该是什么。
        :param backend: 可选['raw', 'numpy', 'torch', 'torch', 'jittor', 'auto']，分别代表，输出为 list, numpy.ndarray,
            torch.Tensor, torch.Tensor, jittor.Var 类型。若 pad_val 为 None ，该值无意义 。
        :param pad_fn: 指定当前 field 的 pad 函数，传入该函数则 pad_val, dtype, backend 等参数失效。pad_fn 的输入为当前 field 的
            batch 形式。 Collator 将自动 unbatch 数据，然后将各个 field 组成各自的 batch 。pad_func 的输入即为 field 的 batch
            形式，输出将被直接作为结果输出。
        :return: 返回 Collator
        """
        collator = self._get_collator()
        if isinstance(collator, Collator):
            collator.set_pad(field_name=field_name, pad_val=pad_val, dtype=dtype, pad_fn=pad_fn, backend=backend)
            return collator
        else:
            raise ValueError(f"Only when the collate_fn is a fastNLP Collator, set_pad() is allowed.")

    def _get_collator(self):
        """
        如果 collate_fn 是 Collator 对象，得到该对象。如果没有的话，返回 None

        :return:
        """
        collator = None
        if hasattr(self.collate_fn, '__wrapped__') and isinstance(self.collate_fn.__wrapped__, Collator):
            collator = self.collate_fn.__wrapped__
        elif isinstance(self.collate_fn, Collator):
            collator = self.collate_fn
        return collator

    def set_ignore(self, *field_names) -> Collator:
        """
        如果有的内容不希望输出，可以在此处进行设置，被设置的 field 将在 batch 的输出中被忽略。
        Example::

            collator.set_ignore('field1', 'field2')

        :param field_names: 需要忽略的 field 的名称。如果 Dataset 的 __getitem__ 方法返回的是 dict 类型的，则可以直接使用对应的
            field 的 key 来表示，如果是 nested 的 dict，可以使用元组来表示，例如 {'a': {'b': 1}} 中的使用 ('a', 'b'); 如果
            __getitem__ 返回的是 Sequence 类型的，则可以使用 '_0', '_1' 表示序列中第 0 或 1 个元素。
        :return: 返回 Collator 自身
        """
        collator = self._get_collator()
        if isinstance(collator, Collator):
            collator.set_ignore(*field_names)
            return collator
        else:
            raise ValueError(f"Only when the collate_fn is a fastNLP Collator, set_ignore() is allowed.")

    def get_batch_indices(self) -> List[int]:
        """
        获取当前 ``batch`` 中每条数据对应的索引。

        :return: 当前 ``batch`` 数据的索引；
        """
        return self.cur_batch_indices


def prepare_torch_dataloader(ds_or_db,
                             batch_size: int = 16,
                             shuffle: bool = False,
                             sampler: Union["Sampler[int]", ReproducibleSampler, UnrepeatedSampler] = None,
                             batch_sampler: Union["Sampler[Sequence[int]]", ReproducibleBatchSampler] = None,
                             num_workers: int = 0, collate_fn: Union[Callable, str, None] = 'auto',
                             pin_memory: bool = False, drop_last: bool = False,
                             timeout: float = 0, worker_init_fn: Optional[Callable] = None,
                             multiprocessing_context=None, generator=None, prefetch_factor: int = 2,
                             persistent_workers: bool = False,
                             non_train_sampler: Union["Sampler[int]", ReproducibleSampler, UnrepeatedSampler] = None,
                             non_train_batch_size: int = 16) \
        -> Union[TorchDataLoader, Dict[str, TorchDataLoader], Sequence[TorchDataLoader]]:
    """
    ``prepare_torch_dataloader`` 的功能是将输入的单个或多个 dataset 同时转为 ``TorchDataloader``对象， 详见 :class: `~fastNLP.core.dataloaders.TorchDataLoader`。
    根据 ds_or_db 的类型 ``[DataSet, DataBundle,Sequence[Dataset], Dict[name, Dataset]]`` 不同而有不同返回结果, 具体如下:

        * 当 ds_or_db 为 ``DataSet``时，``prepare_torch_dataloader`` 会将使用的除了 non_train_batch_size 和 non_train_sampler 以外的参数来
        帮你实例化一个 ``TorchDataLoader`` 对象并返回该对象。 详见:class: `~fastNLP.core.dataloaders.TorchDataLoader`。
        * 当 ds_or_db 为 :class:`~fastNLP.io.DataBundle` 时，``prepare_torch_dataloader`` 会遍历 ``DataBundle`` 的数据集的 key-value
        来创建不同的 ``TorchDataLoader`` 对象；当 key 中包含'train'字符串时，``prepare_torch_dataloader`` 默认该 value 为 train 数据集，
        会将 batch_size 和 sampler 作为参数，其他 key 不包含 'train' 字符串的数据集则使用 non_train_size 和 non_train_sampler 作为参数。
        最终根据 ``key: TorchDataLoader`` 组成 ``Dict[key, TorchDataLoader]`` 的字典返回。
        * 当 ds_or_db 为 ``Dict[str, DataSet]`` 字典类型时， ``prepare_torch_dataloader`` 会遍历 该 dict 的的 key-value 来创建不同的
        ``TorchDataLoader`` 对象；当 key 中包含'train'字符串时，``prepare_torch_dataloader`` 默认该 value 为 train 数据集，会将 batch_size 和 sampler 作为参数，
        其他 key 不包含 'train' 字符串的数据集则使用 non_train_size 和 non_train_sampler 作为参数。最终根据  ``key: TorchDataLoader`` 组成
         ``Dict[key, TorchDataLoader]`` 的字典返回。
        * 当 ds_or_db 为 ``Sequence[Dataset]`` 数据类型时， prepare_torch_dataloader 会将 Sequence[0] 的数据集默认为 train 数据集对待，
        会将 batch_size 和 sampler 作为参数， 而 Sequence[1:] 数据集均视为非 train 数据集对待，使用 non_train_size 和 non_train_sampler 作为参数。
        最终将所有实例化好的 ``TorchDataLoader`` 组成 ``Sequence[TorchDataLoader]`` 返回。

    :param ds_or_db: 实现 __getitem__() 和 __len__() 的对象；或这种对象的序列；或字典。其取值只能为 ``[DataSet, DataBundle,
     Sequence[DataSet], Dict[str, DataSet]]``.

        * ds_or_db 为  :class: `~fastNLP.core.dataset.DataSet`，返回值为:class: `~fastNLP.core.dataloaders.TorchDataLoader`
        * ds_or_db 为 :class: `~fastNLP.io.DataBundle`, 返回值为 ``Dict[str, TorchDataLoader]`` 的字典
        * ds_or_db 为 ``Sequence[DataSet]`` 序列， 返回值为 ``Sequence[TorchDataLoader]`` 的序列
        * ds_or_db 为 ``Dict[str, DataSet]`` 字典， 返回值也为 ``Dict[str, TorchDataLoader]`` 的字典

    :param batch_size: 批次大小，默认为 ``16`` 且当 batch_sampler 为 None 有效。
    :param non_train_batch_size: 非 'train' 数据集的 ``TorchDataLoader`` 批次大小，默认为 ``16`` 且当 batch_sampler 为 None 有效。
    :param shuffle: 是否打乱数据集， 默认为 ``False``。
    :param sampler: 实现了 __len__() 和 __iter__() 的实例化对象，其 __iter__() 方法每次都会返回 dataset 的一个下标 index ，
    默认为None， 当其不为 None 时， shuffle 参数无效。
    :param non_train_sampler: 非 'train' 数据集的的实现了 __len__() 和 __iter__() 的实例化对象，其 __iter__() 方法每次都会返回 dataset 的一个下标 index ，
    默认为None， 当其不为 None 时， shuffle 参数无效。
    :param batch_sampler: 实现了 __len__() 和 __iter__() 的实例化对象，，其__iter__() 方法每次都会返回一个 List 对象， List中的值为
    dataset 的下标 index ；默认为 None，当其不为 None 时，bacth_size, sampler, shuffle 参数均失效。
    :param num_workers: 当 ``num_workers > 0`` 时, ``TorchDataLoader`` 会开启 num_workers 个子进程来处理数据， 可以加快
    数据处理速度，但同时也消耗大量内存。 当 ``num_workers=0`` 时， 不开启子进程。 默认为 ``0``。
    :param collate_fn: 用于从 dataset 取到的一个 batch 数据进行打包处理的 Callable 函数，其值应该为以下三个: ``[None, "auto", Callable]``.

        *  callate_fn 为 'None' 时，需要注意的是此时传进来的 datset 类型不能为 :class:`~fastNLP.core.dataset.DataSet` , 当 collate_fn 为 ``None`` 时，
         ``TorchDataLoader`` 调用默认的 torch 框架的 ``DataLoader`` 自带的 `default_collate_fn` 作为 callate_fn 的默认值， 其无法处理
         :class:`~fastNLP.core.dataset.DataSet` 的dataset对象。
        * callate_fn 为 ``'auto'`` 时，`TorchDataLoader`` 使用 :class:`~fastNLP.core.collators.Collator` 作为 collate_fn 的默认值。
        此时可以配套使用 ``TorchDataLoader`` 的 ``set_pad`` 和 ``set_ignore`` 方法来设置 pad_val 或 忽略某个 field 的检测。
        * `collate_fn 为 ``Callable`` 时， 该 Callable 函数应当接受一个 batch 参数作为输入， batch 是一个 List 对象且 List 中的每一条数据都是
        dataset 的一条数据；该 Callable 函数还应当返回一个对象。

    :param pin_memory: 如果其为 ``True``, 那么 ``TorchDataLoader`` 会在返回数据张量之前将其 copy 到 cud a的 pin memory 中。
    :param drop_last: 当 ``drop_last=True`` 时，``TorchDataLoader`` 会扔掉最后一个长度小于 ``batch_size`` 的 batch 数据;
    若 ``drop_last=False`` , 则会返回该 batch 数据。 默认为 ``False`` 。
    :param timeout: 子进程的输出队列获取数据的超时值
    :param worker_init_fn: init 函数，如果不设置为 None ,则将会在每个子进程初始化时调用该函数。
    :param multiprocessing_context: 多进程的上下文环境
    :param generator: 如果其不为 ``None``, 将会使用 RandomSampler 去生成随机的 index 且会为每个子进程生成一个``base_seed``
    :param prefetch_factor: 每个 worker 提前装载的 samples 数量。``2``意味着在所有的进程中会有 2*num_workers 的数据被预取。默认值为 ``2`` .
    :param persistent_workers: 如果其为 ``True``, ``TorchDataLoader`` 在迭代完一次 dataset 后不会关闭所有进程。默认为 ``False``

    """

    from fastNLP.io import DataBundle
    if isinstance(ds_or_db, DataSet):
        dl = TorchDataLoader(dataset=ds_or_db, batch_size=batch_size,
                             shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                             num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                             drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                             multiprocessing_context=multiprocessing_context, generator=generator,
                             prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                             )
        return dl

    elif isinstance(ds_or_db, DataBundle):
        dl_bundle = {}
        for name, ds in ds_or_db.iter_datasets():
            if 'train' in name:
                dl_bundle[name] = TorchDataLoader(dataset=ds, batch_size=batch_size,
                                                  shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                                  drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                                                  multiprocessing_context=multiprocessing_context, generator=generator,
                                                  prefetch_factor=prefetch_factor,
                                                  persistent_workers=persistent_workers,
                                                  )
            else:
                dl_bundle[name] = TorchDataLoader(dataset=ds,
                                                  batch_size=non_train_batch_size if non_train_batch_size else batch_size,
                                                  shuffle=shuffle,
                                                  sampler=non_train_sampler if non_train_sampler else sampler,
                                                  batch_sampler=batch_sampler,
                                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                                  drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                                                  multiprocessing_context=multiprocessing_context, generator=generator,
                                                  prefetch_factor=prefetch_factor,
                                                  persistent_workers=persistent_workers,
                                                  )
        return dl_bundle

    elif isinstance(ds_or_db, Sequence):
        dl_bundle = []
        for idx, ds in enumerate(ds_or_db):
            if idx > 0:
                batch_size = non_train_batch_size if non_train_batch_size else batch_size
                sampler = non_train_sampler if non_train_sampler else sampler
            dl_bundle.append(
                TorchDataLoader(dataset=ds, batch_size=batch_size,
                                shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                                multiprocessing_context=multiprocessing_context, generator=generator,
                                prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                                )
            )
        return dl_bundle

    elif isinstance(ds_or_db, Mapping):
        dl_bundle = {}
        for name, ds in ds_or_db.items():
            if 'train' in name:
                dl_bundle[name] = TorchDataLoader(dataset=ds, batch_size=batch_size,
                                                  shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                                  drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                                                  multiprocessing_context=multiprocessing_context, generator=generator,
                                                  prefetch_factor=prefetch_factor,
                                                  persistent_workers=persistent_workers,
                                                  )
            else:
                dl_bundle[name] = TorchDataLoader(dataset=ds,
                                                  batch_size=non_train_batch_size if non_train_batch_size else batch_size,
                                                  shuffle=shuffle,
                                                  sampler=non_train_sampler if non_train_sampler else sampler,
                                                  batch_sampler=batch_sampler,
                                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                                  drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                                                  multiprocessing_context=multiprocessing_context, generator=generator,
                                                  prefetch_factor=prefetch_factor,
                                                  persistent_workers=persistent_workers,
                                                  )

        return dl_bundle
    else:
        raise ValueError(f"ds_or_db: {ds_or_db} must be fastnlp dataset or data_bundle or sequence or mapping!")
