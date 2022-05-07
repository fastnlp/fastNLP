__all__ = [
    'TorchDataLoader',
    'prepare_torch_dataloader'
]

from typing import Optional, Callable, Sequence, Union, Tuple, Dict, Mapping, List
import inspect

from fastNLP.core.dataset import DataSet
from fastNLP.core.collators import Collator
from fastNLP.core.dataloaders.utils import indice_collate_wrapper
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleSampler, UnrepeatedSampler, RandomSampler

if _NEED_IMPORT_TORCH:
    from torch.utils.data import DataLoader, Sampler
else:
    from fastNLP.core.utils.dummy_class import DummyClass as DataLoader


class _FDataSet:
    """
    对Dataset的封装，主要是修改dataset的__getitem__函数，增加返回下标idx，值得注意的是dataset需要实现__getattribute__函数才能在_FDataset
    中调用dataset的方法
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
    提供给使用pytorch框架的DataLoader函数，若是配套使用FastNLP的dataset则可以自动使用AutoCollate函数对数据进行自动padding操作，用户也可以通过
    提供的方法调节设置collate_fn的若干参数。
    """

    def __init__(self, dataset, batch_size: int = 1,
                 shuffle: bool = False, sampler: Union["Sampler[int]", ReproducibleSampler, UnrepeatedSampler] = None,
                 batch_sampler: Union["Sampler[Sequence[int]]", ReproducibleBatchSampler] = None,
                 num_workers: int = 0, collate_fn: Union[Callable, str, None] = 'auto',
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[Callable] = None,
                 multiprocessing_context=None, generator=None, prefetch_factor: int = 2,
                 persistent_workers: bool = False, **kwargs) -> None:
        """

        :param dataset: 实现了__getitem__和__len__的数据容器
        :param batch_size: 批次大小，当batch_sampler为None生效
        :param shuffle: 是否打乱数据集
        :param sampler: sampler实例化对象
        :param batch_sampler: batch_sampler实例化对象，其能迭代返回一个list的index数据
        :param num_workers: 进程的数量，当num_worker=0时不开启多进程
        :param collate_fn: [None, 'auto', callable] 对取得到的数据进行打包的callable函数
        :param pin_memory:
        :param drop_last: 是否去掉最后一个不符合batch_size的数据
        :param timeout:
        :param worker_init_fn:
        :param multiprocessing_context:
        :param generator:
        :param prefetch_factor:
        :param persistent_workers:
        """
        if isinstance(dataset, DataSet) and collate_fn is None:
            raise ValueError("When use FastNLP DataSet, collate_fn must be not None")

        if not isinstance(dataset, _FDataSet):
            dataset = _FDataSet(dataset)

        if sampler is None and batch_sampler is None:
            sampler = RandomSampler(dataset, shuffle=shuffle)
            shuffle = False
        if isinstance(collate_fn, str):
            if collate_fn == 'auto':
                if isinstance(dataset.dataset, DataSet):  # 使用了 fastnlp dataset
                    collate_fn = dataset.dataset.collator
                    collate_fn.set_backend(backend="torch")
                else:
                    collate_fn = Collator(backend="torch")
            else:
                raise ValueError(f"collate_fn: {collate_fn} must be 'auto'")

        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                         batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn,
                         pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                         multiprocessing_context=multiprocessing_context, generator=generator,
                         prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers)

        self.cur_batch_indices = None

    def __iter__(self):
        # 如果没有auto_collator 也没有自定义collate_fn， 那么此时采用dataloader自带的collate_fn， 将数据打包即可。
        # if len(self._collate_fn.get_collators()) == 0:
        #     self._collate_fn.add_collator(self.collate_fn)
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
        :param backend: 可选['raw', 'numpy', 'torch', 'paddle', 'jittor', 'auto']，分别代表，输出为 list, numpy.ndarray,
            torch.Tensor, paddle.Tensor, jittor.Var 类型。若 pad_val 为 None ，该值无意义 。
        :param pad_fn: 指定当前 field 的 pad 函数，传入该函数则 pad_val, dtype, backend 等参数失效。pad_fn 的输入为当前 field 的
            batch 形式。 Collator 将自动 unbatch 数据，然后将各个 field 组成各自的 batch 。pad_func 的输入即为 field 的 batch
            形式，输出将被直接作为结果输出。
        :return: 返回 Collator 自身
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
        获取当前 batch 的 idx

        :return:
        """
        return self.cur_batch_indices


def prepare_torch_dataloader(ds_or_db: Union[DataSet, Sequence[DataSet], Mapping[str, DataSet]],
                             batch_size: int = 16,
                             shuffle: bool = True,
                             sampler: Union["Sampler[int]", ReproducibleSampler, UnrepeatedSampler] = None,
                             batch_sampler: Union["Sampler[Sequence[int]]", ReproducibleBatchSampler] = None,
                             num_workers: int = 0, collate_fn: Union[str, Callable, None] = 'auto',
                             pin_memory: bool = False, drop_last: bool = False,
                             timeout: float = 0, worker_init_fn: Optional[Callable] = None,
                             multiprocessing_context=None, generator=None, prefetch_factor: int = 2,
                             persistent_workers: bool = False, non_train_sampler: Optional["Sampler[int]"] = None,
                             non_train_batch_size: int = 16) \
        -> Union[TorchDataLoader, Dict[str, TorchDataLoader], Sequence[TorchDataLoader]]:
    """
    传入dataset或者data_bundle后，将其处理返回相对应的FdataLoader实例化对象

    :param input_fields:
    :param ds_or_db: dataset或者data_bundle
    :param batch_size: 批次大小，当batch_sampler为None生效
    :param shuffle: 是否打乱数据集
    :param sampler: sampler实例化对象
    :param batch_sampler: batch_sampler实例化对象，其能迭代返回一个list的index数据
    :param num_workers: 进程的数量，当num_worker=0时不开启多进程
    :param collate_fn: ['auto', None, callable]对取得到的数据进行打包的callable函数
    :param pin_memory:
    :param drop_last: 是否去掉最后一个不符合batch_size的数据
    :param timeout:
    :param worker_init_fn:
    :param multiprocessing_context:
    :param generator:
    :param prefetch_factor:
    :param persistent_workers:
    :param non_train_sampler: 非 'train' 数据使用的 Sampler, 以及Sequence的第二个以上的ds使用的 Sampler
    :param non_train_batch_size:
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
                dl_bundle[name] = TorchDataLoader(dataset=ds, batch_size=non_train_batch_size,
                                                  shuffle=shuffle, sampler=non_train_sampler,
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
            if idx == 0:
                dl_bundle.append(
                    TorchDataLoader(dataset=ds, batch_size=batch_size,
                                    shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                    num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                    drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                                    multiprocessing_context=multiprocessing_context, generator=generator,
                                    prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                                    )
                )
            else:
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
                dl_bundle[name] = TorchDataLoader(dataset=ds, batch_size=non_train_batch_size,
                                                  shuffle=shuffle, sampler=non_train_sampler,
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
