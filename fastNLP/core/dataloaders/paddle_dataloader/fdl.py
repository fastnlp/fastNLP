"""
``PaddleDataLoader``是专门提供给``paddle``框架的``DataLoader``,其集成了``fastNLP``的``Collator``并对``paddle``的``DataLoader``进行了
封装，使得其具备以下功能：1.``PaddleDataLoader``支持输入的dataset是无框架的，只要实现了``__getitem__``和``__len__``方法即可，当不使用``fastNLP``的
``DataSet``时候也能够自动检测数据的类型并进行padding，只需要将``collate_fn="auto"``即可，例如::

    from fastNLP import PaddleDataLoader
    class MyDataset:
        def __init(self, data_lens=100):
            self.data_lens = 100
        def __getitem__(self, item):
            if item % 2 == 0:
                return {'x':[101, 256, 453], 'y': 0}
            else:
                return {'x': [101, 200], 'y': 1}
        def __len__(self):
            return self.data_lens
    dataset = MyDataset()
    paddle_dl = PaddleDataLoader(dataset, collate_fn="auto")
    for batch in paddle_dl:
        ...

2.当设置``collate_fn="auto"``时，``PaddleDataLoader``会调用fastNLP的Collator对数据进行自动pad处理，此时可以调用``set_pad``和``set_ignore``方法
来设置field的pad_val或者忽略某个field的pad操作。
.. note::
    当传入的dataset为fastNLP的DataSet时，collate_fn不能为None。默认可以是"auto"或者自定义callable函数。

"""

__all__ = [
    'PaddleDataLoader',
    'prepare_paddle_dataloader'
]

from typing import Callable, List, Optional, Union, Dict, Sequence
from copy import deepcopy

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE

if _NEED_IMPORT_PADDLE:
    from paddle.io import DataLoader, Dataset, Sampler
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Dataset
    from fastNLP.core.utils.dummy_class import DummyClass as DataLoader
    from fastNLP.core.utils.dummy_class import DummyClass as Sampler

from fastNLP.core.collators.collator import Collator
from fastNLP.core.dataloaders.utils import indice_collate_wrapper
from fastNLP.core.dataset import DataSet as FDataSet
from fastNLP.core.samplers import ReproducibleBatchSampler, RandomBatchSampler


class _PaddleDataset(Dataset):
    """
    对用户传的dataset进行封装，以便PaddleDataLoader能够支持使用自定义的dataset
    """

    def __init__(self, dataset) -> None:
        super(_PaddleDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, item):
        return (item, self.dataset[item])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, item):
        try:
            self.dataset.__getattribute__(item)
        except Exception as e:
            raise e


class PaddleDataLoader(DataLoader):
    """
    提供给``paddle``框架使用的``DataLoader``函数，``PaddleDataLoader``提供了``Collator``的功能，用户可以通过设置``collate_fn="auto"``来
    使用，并可以配套使用``set_pad``和``set_ignore``方法设置p``ad_val``和忽略某个field的pad操作。
    """

    def __init__(self, dataset, feed_list=None, places=None,
                 return_list: bool = True, batch_sampler=None,
                 batch_size: int = 1, shuffle: bool = True,
                 drop_last: bool = False, collate_fn: Union[str, Callable, None] = 'auto',
                 num_workers: int = 0, use_buffer_reader: bool = True,
                 use_shared_memory: bool = True, timeout: int = 0,
                 worker_init_fn: Callable = None, persistent_workers=False) -> None:
        """

        :param dataset: 实现了__getitem__和__len__的数据容器
        :param feed_list: (list(Tensor)|tuple(Tensor)): feed Tensor list.
            The Tensors should be created by :code:`paddle.static.data()`.
            :attr:`feed_list` must be set if :attr:`return_list` is
            False. Default None.
        :param places: (list(Place)|tuple(Place)|list(str)|optional): a list of Place,
            to put data onto, :attr:`places` can be None, if
            :attr:`places` is None, default place(CPUPlace or CUDAPlace(0))
            will be used. Default None. If ``places`` is list of string,
            the string in the list can be ``cpu``, ``gpu:x`` and ``gpu_pinned``,
            where ``x`` is the index of the GPUs.
        :param return_list: whether the return value on each device is
            presented as a list. If :attr:`return_list=False`, the return
            value on each device would be a dict of str -> Tensor, where
            the key of the dict is the name of each fed Tensors. If
            :attr:`return_list=True`, the return value on each device would
            be a list(Tensor). :attr:`return_list` can only be True
            in dynamic graph mode. Default True.
        :param batch_sampler: 实现了``__iter__``和``__len__``方法的实例化对象，它的功能是根据dataset生成数据indices并组成一个batch数据。
        :param batch_size: dataloader每次获得数据的批次大小
        :param shuffle: 是否将数据打乱，若``shuffle=True``则会将dataset打乱；若否则什么也不做。
        :param drop_last: 当``drop_last=True``时，``PaddleDataLoader``会扔掉最后一个不能组成``batch_size``大小的batch数据;
        若``drop_last=False``, 则什么也不做。
        :param collate_fn:用来对从dataset取到的数据进行打包处理成batch的callable函数，其值应该为一下三个:``[None, "auto", callable]``.

            * ``callate_fn=None``时，第一点值得注意的是此时传进来的datset不能为``fastNLP``的dataset,采用fastNLP的dataset时，``collate_fn``不能为``None``;
            第二点注意的是此时``PaddleDataLoader``会调用默认的`default_collate_fn`函数对sampler到的数据进行简单打包，组成一个batch返回。`
            * ``callate_fn="auto"``时，``PaddleDataLoader``会自动调用``fastNLP``自带的``Collator``，其会自动检测dataset的每个``field``,
            并判断是否能够pad处理，若能则会自动进行pad操作，默认``pad_val=0``。若想要更改其值，可调用``set_pad``方法;若不想自动pad某个field，
            可以调用``set_ignore``方法忽略某个field。
            * ``callate_fn=callable``时，callable函数是用户自定义的callate_fn函数，此时``PaddleDataLoader``会调用传进来的callable函数对
            数据进行打包处理并返回。值得注意的是用户自定义的callable函数的输入为batch,batch为list类型数据，其中batch的每一条数据都为dataset的一条数据。

        :param num_workers: 开启多进程的数量，当``num_workers=0``时不开启多进程
        :param use_buffer_reader: 是否开启buffer_reader。如果``use_buffer_reader=True``，那么``PaddleDataLoader``将会异步的预取下一个batch的
            数据，因此它将会加快数据传输的速度，但是将会占用更多的内存或者显存。默认值是``True``。如果``use_buffer_reader=False``,那么什么也不错
        :param use_shared_memory: 是否使用共享内存。当``use_shared_memory=True``时，将采用共享内存来加快将数据放进进程队列。建议仅当计算机上的
            共享空间足够大时。（例如Linux上的/dev/shm/空间足够大）共享内存仅在多进程模式（num_workers>0）下生效。
        :param timeout: 从子进程的输出队列获取数据的超时值
        :param worker_init_fn: init函数，如果不设置为None,则将会在每个子进程初始化时调用该函数。
        :param persistent_workers:

        """
        # FastNLP Datset, collate_fn not None
        if isinstance(dataset, FDataSet) and collate_fn is None:
            raise ValueError("When use FastNLP DataSet, collate_fn must be not None")

        if not isinstance(dataset, _PaddleDataset):
            dataset = _PaddleDataset(dataset)

        if batch_sampler is None:
            batch_sampler = RandomBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle,
                                               drop_last=drop_last)
            batch_size = 1
            shuffle = False
            drop_last = False

        if isinstance(collate_fn, str):
            if collate_fn == 'auto':
                if isinstance(dataset.dataset, FDataSet):
                    collate_fn = deepcopy(dataset.dataset.collator)
                    collate_fn.set_backend(backend="paddle")
                else:
                    collate_fn = Collator(backend="paddle")

            else:
                raise ValueError(f"collate_fn: {collate_fn} must be 'auto'")

        super(PaddleDataLoader, self).__init__(dataset=dataset, feed_list=feed_list, places=places,
                                               return_list=return_list, batch_sampler=batch_sampler,
                                               batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                               collate_fn=collate_fn, num_workers=num_workers,
                                               use_buffer_reader=use_buffer_reader, use_shared_memory=use_shared_memory,
                                               timeout=timeout, worker_init_fn=worker_init_fn,
                                               persistent_workers=persistent_workers)

        # _collate_fn = _MultiCollator(AutoCollator(as_numpy=True))
        # if collate_fn is not None:
        #     _collate_fn.add_collator(collate_fn)
        # self._collate_fn = _collate_fn
        self.cur_batch_indices = None

    def __iter__(self):
        # 如果没有auto_collator 也没有自定义collate_fn， 那么此时采用dataloader自带的collate_fn， 将数据打包即可。
        # if len(self._collate_fn.get_collators()) == 0:
        #     self._collate_fn.add_collator(default_collate_fn)
        # self._collate_fn = default_collate_fn
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
        获取当前 ``batch`` 中每条数据对应的索引。

        :return: 当前 ``batch`` 数据的索引；
        """
        return self.cur_batch_indices


def prepare_paddle_dataloader(ds_or_db, feed_list=None, places=None,
                              return_list: bool = True,
                              batch_sampler: Union["Sampler[Sequence[int]]", ReproducibleBatchSampler] = None,
                              train_batch_size: int = 1, shuffle: bool = False,
                              drop_last: bool = False, collate_fn: Union[Callable, str, None] = 'auto',
                              num_workers: int = 0, use_buffer_reader: bool = True,
                              use_shared_memory: bool = True, timeout: int = 0,
                              worker_init_fn: Callable = None, persistent_workers=False,
                              non_train_batch_size: int = 16) \
        -> Union[Sequence[PaddleDataLoader], Dict[str, PaddleDataLoader], PaddleDataLoader]:
    from fastNLP.io.data_bundle import DataBundle
    if isinstance(ds_or_db, Dataset):
        dl = PaddleDataLoader(ds_or_db, feed_list=feed_list, places=places, return_list=return_list,
                              batch_sampler=batch_sampler, batch_size=train_batch_size, shuffle=shuffle,
                              drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                              use_shared_memory=use_shared_memory, use_buffer_reader=use_buffer_reader,
                              timeout=timeout, worker_init_fn=worker_init_fn, persistent_workers=persistent_workers)
        return dl
    elif isinstance(ds_or_db, DataBundle):
        dl_bundle = {}
        for name, ds in ds_or_db.iter_datasets():
            if 'train' in name:
                dl_bundle[name] = PaddleDataLoader(ds, feed_list=feed_list, places=places,
                                                   return_list=return_list,
                                                   batch_sampler=batch_sampler, batch_size=train_batch_size,
                                                   shuffle=shuffle,
                                                   drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                                                   use_shared_memory=use_shared_memory,
                                                   use_buffer_reader=use_buffer_reader,
                                                   timeout=timeout, worker_init_fn=worker_init_fn,
                                                   persistent_workers=persistent_workers)
            else:
                dl_bundle[name] = PaddleDataLoader(ds, feed_list=feed_list, places=places,
                                                   return_list=return_list,
                                                   batch_sampler=batch_sampler, batch_size=non_train_batch_size,
                                                   shuffle=shuffle,
                                                   drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                                                   use_shared_memory=use_shared_memory,
                                                   use_buffer_reader=use_buffer_reader,
                                                   timeout=timeout, worker_init_fn=worker_init_fn,
                                                   persistent_workers=persistent_workers)
        return dl_bundle
    elif isinstance(ds_or_db, Sequence):
        ds_seq = []
        for ds in ds_or_db:
            dl = PaddleDataLoader(ds, feed_list=feed_list, places=places, return_list=return_list,
                                  batch_sampler=batch_sampler, batch_size=train_batch_size, shuffle=shuffle,
                                  drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                                  use_shared_memory=use_shared_memory, use_buffer_reader=use_buffer_reader,
                                  timeout=timeout, worker_init_fn=worker_init_fn, persistent_workers=persistent_workers)
            ds_seq.append(dl)
        return ds_seq

    elif isinstance(ds_or_db, Dict):
        ds_dict = {}
        for name, ds in ds_or_db.items():
            if 'train' in name:
                dl = PaddleDataLoader(ds, feed_list=feed_list, places=places, return_list=return_list,
                                      batch_sampler=batch_sampler, batch_size=train_batch_size, shuffle=shuffle,
                                      drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                                      use_shared_memory=use_shared_memory, use_buffer_reader=use_buffer_reader,
                                      timeout=timeout, worker_init_fn=worker_init_fn,
                                      persistent_workers=persistent_workers)
            else:
                dl = PaddleDataLoader(ds, feed_list=feed_list, places=places, return_list=return_list,
                                      batch_sampler=batch_sampler, batch_size=non_train_batch_size, shuffle=shuffle,
                                      drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                                      use_shared_memory=use_shared_memory, use_buffer_reader=use_buffer_reader,
                                      timeout=timeout, worker_init_fn=worker_init_fn,
                                      persistent_workers=persistent_workers)
            ds_dict[name] = dl
        return ds_dict
    else:
        raise ValueError(f"ds_or_db: {ds_or_db} must be fastnlp dataset or data_bundle or sequence or mapping!")
