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
from ..utils import _match_param, HasLenGetitemType


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
            return self.dataset.__getattribute__(item)
        except Exception as e:
            raise e


class PaddleDataLoader(DataLoader):
    """
    ``PaddleDataLoader`` 是专门提供给 ``paddle`` 框架的 ``DataLoader`` ,其集成了 ``fastNLP`` 的 ``Collator`` ，
    具体详见 :class:`~fastNLP.core.collators.Collator`， 并对 ``paddle`` 的 ``DataLoader`` 进行了
    封装，使得其具备以下功能：
    
    1. ``PaddleDataLoader`` 支持输入的 dataset 是无框架的，只要实现了 __getitem__() 和 __len__() 的对象即可，
       当不使用  :class:`~fastNLP.core.dataset.DataSet` 时也不需要传入 collate_fn, 只要只需要将 ``collate_fn='auto'`` 就能够自动
       探测数据的类型并判断能否 pad 。此时可以调用 ``set_pad`` 和 ``set_ignore`` 方法来设置 field 的 pad_val 或者忽略某个 field 的 pad 操作。
    
        Example::

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
            paddle_dl = PaddleDataLoader(dataset, collate_fn='auto')
            for batch in paddle_dl:
                ...

    2.当 collate_fn 为 ``None`` 时，``PaddleDataLoader`` 默认使用 ``paddle`` 自带的 ``default_collate_fn`` 作为 collate_fn 的值

        .. note::
            当传入的dataset为fastNLP的DataSet时，collate_fn不能为None。默认可以是"auto"或者自定义callable函数。

    3. 当 collate_fn 为 :class:`Callable` 时，该 Callable 函数应当接受一个 batch 参数作为输入， batch 是一个 List 对象且 List 中的每一条数据都是
       dataset 的一条数据；该 Callable 函数还应当返回一个对象。

    :param dataset: 实现了 __getitem__() 和 __len__() 的对象。
    :param feed_list: feed Tensor list.
        这个张量能被 ``paddle.static.data`` 创建。 如果 :attr:`return_list` 是 ``False``, 那么 :attr:`feed_list`
        应该被设置。 默认为 ``None `` 。
    :param places: 将数据放进的一个 list 的 place。 :attr:`places` 能为 None.
        如果 :attr:`places` 为 None， 默认放在 CPUPlace 或者 CUDAPlace(0) 设备上。 如果 ``places`` 是一个 list 类型的 字符串， 那么字符串
        可以是 ``cpu`` , ``gpu:x`` 或者 ``gpu_pinned`` ， 其中 ``x`` 是 gpu 的下标。
    :param return_list: 每个设备上的返回值是否为以列表形式显示。 如果 :attr:`return_list=False`,
        每个设备上的返回值值为 str -> Tensor 的 dict， 其中 dict 的 key 为每个 fed Tensors 的名字。
        如果 :attr:`return_list=True`， 每个设备上的返回值值为 list(Tensor)。 :attr:`return_list` 只能在动态图情况下设置为 ``True`` .
        默认值为 ``True`` 。
    :param batch_sampler: 实现了 __len__() 和 __iter__() 的实例化对象，，其__iter__() 方法每次都会返回一个 List 对象， List中的值为
        dataset 的下标 index ；默认为 ``None``，当其不为 ``None`` 时，``bacth_size``, ``shuffle`` 参数均失效。
    :param batch_size: 批次大小，默认为 ``16`` 且当 ``batch_sampler`` 为 None 有效。
    :param shuffle: 是否打乱数据集， 默认为 ``None``, 如果传入的 ``ds_or_db`` 可以判断出哪个是 ``'train'`` 则设置其 shuffle 为 ``True`` ，
        其它的为 False 。
    :param drop_last: 当 ``drop_last=True`` 时，``PaddleDataLoader`` 会扔掉最后一个长度小于 ``batch_size`` 的 batch 数据;
        若 ``drop_last=False`` , 则会返回该 batch 数据。 默认为 ``False`` 。
    :param collate_fn: 用于从 dataset 取到的一个 batch 数据进行打包处理的 Callable 函数，其值应该为以下三个: ``[None, "auto", Callable]``.

        * callate_fn 为 ``None`` 时，需要注意的是此时传进来的 datset 类型不能为 :class:`~fastNLP.core.dataset.DataSet` , 当 collate_fn 为 ``None`` 时，
        ``PaddleDataLoader`` 调用默认的 Paddle 框架的 ``DataLoader`` 自带的 `default_collate_fn` 作为 callate_fn 的默认值， 其无法处理
        :class:`~fastNLP.core.dataset.DataSet` 的dataset对象。
        * callate_fn 为 ``'auto'`` 时，``PaddleDataLoader`` 使用 :class:`~fastNLP.core.collators.Collator` 作为 collate_fn 的默认值。
        此时可以配套使用 ``PaddleDataLoader`` 的 ``set_pad`` 和 ``set_ignore`` 方法来设置 pad_val 或 忽略某个 field 的检测。
        * collate_fn 为 :class:`Callable` 时， 该 Callable 函数应当接受一个 batch 参数作为输入， batch 是一个 List 对象且 List 中的每一条数据都是
        dataset 的一条数据；该 Callable 函数还应当返回一个对象。

    :param num_workers: 当 ``num_workers > 0`` 时, ``PaddleDataLoader`` 会开启 ``num_workers`` 个子进程来处理数据， 可以加快
        数据处理速度，但同时也消耗大量内存。 当 ``num_workers=0`` 时， 不开启子进程。 默认为 ``0``。
    :param use_buffer_reader: 是否开启 buffer_reader 。如果 ``use_buffer_reader=True`` ，那么 ``PaddleDataLoader`` 会异步地预取下一个 batch 的
        数据，因此它将会加快数据传输的速度，但是将会占用更多的内存或者显存。默认值是 ``True``。
    :param use_shared_memory: 是否使用共享内存。当 ``use_shared_memory=True`` 时，将采用共享内存来加快将数据放进进程队列。建议仅当计算机上的
        共享空间足够大时。（例如 Linux 上的 /dev/shm/ 空间足够大）共享内存仅在多进程模式（ ``num_workers>0`` ）下生效。
    :param timeout: 从子进程的输出队列获取数据的超时值
    :param worker_init_fn: init 函数，如果不设置为 None ,则将会在每个子进程初始化时调用该函数。
    :param persistent_workers: 如果其为 ``True``, ``PaddleDataLoader`` 在迭代完一次 dataset 后不会关闭所有进程。默认为 ``False`` 
    """

    def __init__(self, dataset, feed_list=None, places=None,
                 return_list: bool = True, batch_sampler=None,
                 batch_size: int = 16, shuffle: bool = False,
                 drop_last: bool = False, collate_fn: Union[str, Callable, None] = 'auto',
                 num_workers: int = 0, use_buffer_reader: bool = True,
                 use_shared_memory: bool = True, timeout: int = 0,
                 worker_init_fn: Callable = None, persistent_workers=False) -> None:

        # FastNLP Datset, collate_fn not None
        if isinstance(dataset, FDataSet) and collate_fn is None:
            raise ValueError("When use FastNLP DataSet, collate_fn must be not None")

        if not isinstance(dataset, _PaddleDataset):
            dataset = _PaddleDataset(dataset)

        if batch_sampler is None:
            batch_sampler = RandomBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle,
                                               drop_last=drop_last)
        # 因为无论如何传给 DataLoader 的 batch_sampler 都不是 None
        # 所以要恢复默认值防止报错
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

        dl_kwargs = _match_param(PaddleDataLoader.__init__, DataLoader.__init__, DataLoader.__name__)
        if dl_kwargs is None:
            super(PaddleDataLoader, self).__init__(dataset=dataset, feed_list=feed_list, places=places,
                                                   return_list=return_list, batch_sampler=batch_sampler,
                                                   batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                                   collate_fn=collate_fn, num_workers=num_workers,
                                                   use_buffer_reader=use_buffer_reader, use_shared_memory=use_shared_memory,
                                                   timeout=timeout, worker_init_fn=worker_init_fn,
                                                   persistent_workers=persistent_workers)
        else:
            super().__init__(**dl_kwargs)
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

        :param field_name: 需要调整的 field 的名称。如果 :meth:`Dataset.__getitem__` 方法返回的是字典类型，则可以直接使用对应的
            field 的 key 来表示，如果是嵌套字典，可以使用元组表示多层次的 key，例如 ``{'a': {'b': 1}}`` 中可以使用 ``('a', 'b')``;
            如果 :meth:`Dataset.__getitem__` 返回的是 Sequence 类型，则可以使用 ``'_0'``, ``'_1'`` 表示序列中第 **0** 或 **1** 个元素。
            如果该 field 在数据中没有找到，则报错；如果 :meth:`Dataset.__getitem__` 返回的是就是整体内容，请使用 "_single" 。
        :param pad_val: 这个 field 的默认 pad 值。如果设置为 ``None``，则表示该 field 不需要 pad , fastNLP 默认只会对可以 pad 的
            field 进行 pad，所以如果对应 field 本身就不是可以 pad 的形式，可以不需要主动设置为 ``None`` 。如果 ``backend`` 为 ``None``，
            该值无意义。
        :param dtype: 对于需要 pad 的 field ，该 field 数据的 ``dtype`` 。
        :param backend: 可选 ``['raw', 'numpy', 'torch', 'paddle', 'jittor', 'oneflow', 'auto']`` ，分别代表，输出为 :class:`list`, 
            :class:`numpy.ndarray`, :class:`torch.Tensor`, :class:`paddle.Tensor`, :class:`jittor.Var`, :class:`oneflow.Tensor` 类型。
            若 ``pad_val`` 为 ``None`` ，该值无意义 。
        :param pad_fn: 指定当前 field 的 pad 函数，传入该函数则 ``pad_val``, ``dtype``, ``backend`` 等参数失效。``pad_fn`` 的输入为当前 field 的
            batch 形式。 collator 将自动 unbatch 数据，然后将各个 field 组成各自的 batch 。
        :return: 返回使用的 collator
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
        如果有的内容不希望输出，可以在此处进行设置，被设置的 field 将在 batch 的输出中被忽略::

            dataloader.set_ignore('field1', 'field2')

        :param field_names: field_name: 需要调整的 field 的名称。如果 :meth:`Dataset.__getitem__` 方法返回的是字典类型，则可以直接使用对应的
            field 的 key 来表示，如果是嵌套字典，可以使用元组表示多层次的 key，例如 ``{'a': {'b': 1}}`` 中可以使用 ``('a', 'b')``;
            如果 :meth:`Dataset.__getitem__` 返回的是 Sequence 类型，则可以使用 ``'_0'``, ``'_1'`` 表示序列中第 **0** 或 **1** 个元素。
        :return: 返回使用的 collator
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
                              batch_size: int = 16, shuffle: bool = False,
                              drop_last: bool = False, collate_fn: Union[Callable, str, None] = 'auto',
                              num_workers: int = 0, use_buffer_reader: bool = True,
                              use_shared_memory: bool = True, timeout: int = 0,
                              worker_init_fn: Callable = None, persistent_workers=False,
                              non_train_batch_size: int = None) \
        -> Union[Dict[str, PaddleDataLoader], PaddleDataLoader]:
    """
    ``prepare_paddle_dataloader`` 的功能是将输入的单个或多个 dataset 同时转为 ``PaddleDataloader`` 对象， 详见 :class:`~fastNLP.PaddleDataLoader`。
    根据 ds_or_db 的类型 ``[DataSet, DataBundle, Dict[name, Dataset]]`` 不同而有不同返回结果, 具体如下:

        * 当 ds_or_db 为 ``DataSet`` 时，``prepare_paddle_dataloader`` 会将除了 ``non_train_batch_size`` 和 ``non_train_sampler`` 以外的参数来
          帮你实例化一个 ``PaddleDataLoader`` 对象并返回该对象。 详见 :class:`~fastNLP.core.dataloaders.PaddleDataLoader`。
        * 当 ds_or_db 为 :class:`~fastNLP.io.DataBundle` 时，``prepare_paddle_dataloader`` 会遍历 ``DataBundle`` 的数据集的 key-value
          来创建不同的 ``PaddleDataLoader`` 对象；当 key 中包含 ``'train'`` 字符串时，``prepare_Paddle_dataloader`` 默认该 value 为训练数据集，
          会将 ``batch_size`` 和 ``sampler`` 作为参数，其他 key 不包含 ``'train'`` 字符串的数据集则使用 ``non_train_size`` 和 ``non_train_sampler`` 作为参数。
          最终根据 ``key: PaddleDataLoader`` 组成 ``Dict[key, PaddleDataLoader]`` 的字典返回。
        * 当 ds_or_db 为 ``Dict[str, DataSet]`` 字典类型时， ``prepare_paddle_dataloader`` 会遍历 该 dict 的的 key-value 来创建不同的
          ``PaddleDataLoader`` 对象；当 key 中包含 ``'train'`` 字符串时，``prepare_paddle_dataloader`` 默认该 value 为训练数据集，会将 ``batch_size`` 和 ``sampler`` 作为参数，
          其他 key 不包含 ``'train'`` 字符串的数据集则使用 ``non_train_size`` 和 ``non_train_sampler`` 作为参数。最终根据  ``key: PaddleDataLoader`` 组成
          ``Dict[key, PaddleDataLoader]`` 的字典返回。

    :param ds_or_db: 可以有以下三种取值，

        * ds_or_db 为 :class:`~fastNLP.io.DataBundle`, 返回值为 ``Dict[str, TorchDataLoader]`` 的字典；
        * ds_or_db 为 ``Dict[str, DataSet]`` 字典， 返回值为 ``Dict[str, TorchDataLoader]`` 的字典；
        * ds_or_db 为实现了 __getitem__() 和 __len__() 的对象 ，返回值为 :class:`~fastNLP.TorchDataLoader`；

    :param feed_list: feed Tensor list.
        这个张量能被 ``paddle.static.data`` 创建。 如果 :attr:`return_list` 是 ``False``, 那么 :attr:`feed_list`
        应该被设置。 默认为 ``None `` 。
    :param places: 将数据放进的一个 list 的 place。 :attr:`places` 能为 None.
        如果 :attr:`places` 为 None， 默认放在 CPUPlace 或者 CUDAPlace(0) 设备上。 如果 ``places`` 是一个 list 类型的 字符串， 那么字符串
        可以是 ``cpu`` , ``gpu:x`` 或者 ``gpu_pinned`` ， 其中 ``x`` 是 gpu 的下标。
    :param return_list: 每个设备上的返回值是否为以列表形式显示。 如果 :attr:`return_list=False`,
        每个设备上的返回值值为 str -> Tensor 的 dict， 其中 dict 的 key 为每个 fed Tensors 的名字。
        如果 :attr:`return_list=True`， 每个设备上的返回值值为 list(Tensor)。 :attr:`return_list` 只能在动态图情况下设置为 ``True`` .
        默认值为 ``True`` 。
    :param batch_sampler: 实现了 __len__() 和 __iter__() 的实例化对象，，其__iter__() 方法每次都会返回一个 List 对象， List中的值为
        dataset 的下标 index ；默认为 ``None``，当其不为 ``None`` 时，``bacth_size``, ``shuffle`` 参数均失效。
    :param batch_size: 批次大小，默认为 ``16`` 且当 batch_sampler 为 None 有效。
    :param shuffle: 是否打乱数据集， 默认为 ``None``, 如果传入的 ``ds_or_db`` 可以判断出哪个是 ``'train'`` 则设置其 shuffle 为 ``True`` ，
        其它的为 False 。
    :param drop_last: 当 ``drop_last=True`` 时，``PaddleDataLoader`` 会扔掉最后一个长度小于 ``batch_size`` 的 batch 数据;
        若 ``drop_last=False`` , 则会返回该 batch 数据。 默认为 ``False`` 。
    :param collate_fn: 用于从 dataset 取到的一个 batch 数据进行打包处理的 Callable 函数，其值应该为以下三个: ``[None, "auto", Callable]``.

        * callate_fn 为 ``None`` 时，需要注意的是此时传进来的 datset 类型不能为 :class:`~fastNLP.core.dataset.DataSet` , 当 collate_fn 为 ``None`` 时，
          ``PaddleDataLoader`` 调用默认的 Paddle 框架的 ``DataLoader`` 自带的 `default_collate_fn` 作为 callate_fn 的默认值， 其无法处理
          :class:`~fastNLP.core.dataset.DataSet` 的dataset对象。
        * callate_fn 为 ``'auto'`` 时，``PaddleDataLoader`` 使用 :class:`~fastNLP.core.collators.Collator` 作为 collate_fn 的默认值。
          此时可以配套使用 ``PaddleDataLoader`` 的 ``set_pad`` 和 ``set_ignore`` 方法来设置 pad_val 或 忽略某个 field 的检测。
        * collate_fn 为 :class:`Callable` 时， 该 Callable 函数应当接受一个 batch 参数作为输入， batch 是一个 List 对象且 List 中的每一条数据都是
          dataset 的一条数据；该 Callable 函数还应当返回一个对象。

    :param num_workers: 当 ``num_workers > 0`` 时, ``PaddleDataLoader`` 会开启 ``num_workers`` 个子进程来处理数据， 可以加快
        数据处理速度，但同时也消耗大量内存。 当 ``num_workers=0`` 时， 不开启子进程。 默认为 ``0``。
    :param use_buffer_reader: 是否开启 buffer_reader 。如果 ``use_buffer_reader=True`` ，那么 ``PaddleDataLoader`` 会异步地预取下一个 batch 的
        数据，因此它将会加快数据传输的速度，但是将会占用更多的内存或者显存。默认值是 ``True``。
    :param use_shared_memory: 是否使用共享内存。当 ``use_shared_memory=True`` 时，将采用共享内存来加快将数据放进进程队列。建议仅当计算机上的
        共享空间足够大时。（例如 Linux 上的 /dev/shm/ 空间足够大）共享内存仅在多进程模式（ ``num_workers>0`` ）下生效。
    :param timeout: 从子进程的输出队列获取数据的超时值
    :param worker_init_fn: init 函数，如果不设置为 None ,则将会在每个子进程初始化时调用该函数。
    :param persistent_workers: 如果其为 ``True``, ``PaddleDataLoader`` 在迭代完一次 dataset 后不会关闭所有进程。默认为 ``False``

    """
    from fastNLP.io.data_bundle import DataBundle

    if isinstance(ds_or_db, DataBundle):
        dl_bundle = {}
        for name, ds in ds_or_db.iter_datasets():
            if 'train' in name:
                dl_bundle[name] = PaddleDataLoader(ds, feed_list=feed_list, places=places,
                                                   return_list=return_list,
                                                   batch_sampler=batch_sampler, batch_size=batch_size,
                                                   shuffle=True if shuffle is None else shuffle,
                                                   drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                                                   use_shared_memory=use_shared_memory,
                                                   use_buffer_reader=use_buffer_reader,
                                                   timeout=timeout, worker_init_fn=worker_init_fn,
                                                   persistent_workers=persistent_workers)
            else:
                dl_bundle[name] = PaddleDataLoader(ds, feed_list=feed_list, places=places,
                                                   return_list=return_list,
                                                   batch_sampler=batch_sampler,
                                                   batch_size=non_train_batch_size if non_train_batch_size else batch_size,
                                                   shuffle=False if shuffle is None else shuffle,
                                                   drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                                                   use_shared_memory=use_shared_memory,
                                                   use_buffer_reader=use_buffer_reader,
                                                   timeout=timeout, worker_init_fn=worker_init_fn,
                                                   persistent_workers=persistent_workers)
        return dl_bundle

    elif isinstance(ds_or_db, Dict):
        ds_dict = {}
        for name, ds in ds_or_db.items():
            if 'train' in name:
                dl = PaddleDataLoader(ds, feed_list=feed_list, places=places, return_list=return_list,
                                      batch_sampler=batch_sampler, batch_size=batch_size,
                                      shuffle=False if shuffle is None else shuffle,
                                      drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                                      use_shared_memory=use_shared_memory, use_buffer_reader=use_buffer_reader,
                                      timeout=timeout, worker_init_fn=worker_init_fn,
                                      persistent_workers=persistent_workers)
            else:
                dl = PaddleDataLoader(ds, feed_list=feed_list, places=places, return_list=return_list,
                                      batch_sampler=batch_sampler,
                                      batch_size=non_train_batch_size if non_train_batch_size else batch_size,
                                      shuffle=False if shuffle is None else shuffle,
                                      drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                                      use_shared_memory=use_shared_memory, use_buffer_reader=use_buffer_reader,
                                      timeout=timeout, worker_init_fn=worker_init_fn,
                                      persistent_workers=persistent_workers)
            ds_dict[name] = dl
        return ds_dict

    elif isinstance(ds_or_db, HasLenGetitemType):
        dl = PaddleDataLoader(ds_or_db, feed_list=feed_list, places=places, return_list=return_list,
                              batch_sampler=batch_sampler, batch_size=batch_size,
                              shuffle=False if shuffle is None else shuffle,
                              drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                              use_shared_memory=use_shared_memory, use_buffer_reader=use_buffer_reader,
                              timeout=timeout, worker_init_fn=worker_init_fn, persistent_workers=persistent_workers)
        return dl
    else:
        raise ValueError(f"ds_or_db: {ds_or_db} must be fastnlp dataset or data_bundle or mapping!")
