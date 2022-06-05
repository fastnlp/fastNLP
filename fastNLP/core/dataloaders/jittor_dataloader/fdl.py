__all__ = [
    'JittorDataLoader',
    'prepare_jittor_dataloader'
]

from typing import Callable, Optional, List, Union, Dict, Sequence
from copy import deepcopy

import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    from jittor.dataset.utils import collate_batch
    from jittor.dataset import Dataset
else:
    from fastNLP.core.dataset import DataSet as Dataset

from fastNLP.core.collators import Collator
from fastNLP.core.dataloaders.utils import indice_collate_wrapper
from fastNLP.core.dataset import DataSet as FDataSet
from ..utils import HasLenGetitemType


class _JittorDataset(Dataset):
    """
    对用户传的 ``dataset`` 进行封装，以便 ``JittorDataLoader`` 能够支持使用自定义的 ``dataset`` 。
    """

    def __init__(self, dataset) -> None:
        super(_JittorDataset, self).__init__()
        self.dataset = dataset
        self.total_len = len(dataset)

    def __getitem__(self, item):
        if isinstance(item, np.integer):
            item = item.tolist()
        return (item, self.dataset[item])

          
class JittorDataLoader:
    """
    提供给 ``jittor`` 框架使用的 ``DataLoader`` 函数，``JittorDataLoader`` 提供了 ``Collator`` 来自动检测 dataset 的每个 field 是否可 pad，
    若是可 pad 的 field 则自动 pad 到相同长度，否则只会将相同 field 的数据收集组成一个 batch 返回。
    具体详见 :class:`~fastNLP.core.collators.Collator`；用户通过 callte_fn 来控制是否使用该功能， collate_fn 只能为 ``['auto', None, Callable]`` 三种取值。

        * callate_fn 为 ``'auto'`` 时，``JittorDataLoader`` 使用 :class:`~fastNLP.core.collators.Collator` 作为 collate_fn 的取值。
          此时可以配套使用 ``JittorDataLoader`` 的 ``set_pad`` 和 ``set_ignore`` 方法来设置 pad_val 或 忽略某个 field 的检测。
        * callate_fn 为 ``None`` 时， ``JittorDataLoader`` 默认使用 Jittor DataLoader 自带的 collate_fn
        * collate_fn 为 ``Callable`` 时， 该 Callable 函数应当接受一个 batch 参数作为输入， batch 是一个 List 对象且 List 中的每一条数据都是
          dataset 的一条数据；该 Callable 函数还应当返回一个对象。

    """
    def __init__(self, dataset, batch_size: int = 16, shuffle: bool = False,
                 drop_last: bool = False, num_workers: int = 0, buffer_size: int = 512 * 1024 * 1024,
                 stop_grad: bool = True, keep_numpy_array: bool = False, endless: bool = False,
                 collate_fn: Union[None, str, Callable] = "auto") -> None:
        """

        :param dataset: 实现了 __getitem__() 和 __len__() 的对象。
        :param batch_size: 批次大小，默认为 ``16`` 且当 batch_sampler 为 None 有效。
        :param shuffle: 是否打乱数据集， 默认为 ``False``。
        :param drop_last: 当 ``drop_last=True`` 时，``JittorDataLoader`` 会扔掉最后一个长度小于 ``batch_size`` 的 batch 数据;
            若 ``drop_last=False`` , 则会返回该 batch 数据。 默认为 ``False`` 。
        :param num_workers: 当 ``num_workers > 0`` 时, ``JittorDataLoader`` 会开启 num_workers 个子进程来处理数据， 可以加快
            数据处理速度，但同时也消耗大量内存。 当 ``num_workers=0`` 时， 不开启子进程。 默认为 ``0``。
        :param buffer_size: 每个进程占用的内存空间，默认为512M。主要是配合num_workers使用，用户可以自定义每个进程的内存大小。
        :param stop_grad: 是否不使用梯度， 默认 ``True`` 。
        :param keep_numpy_array: 返回的数据是 ``np.array`` 类型而不是 ``jittor.Var`` 类型，默认为 ``False``
        :param endless: 是否让 ``JittorDataLoader`` 无限返回数据，也就是将 dataset 循环使用使得返回数据是没有限制的。默认为 ``False``.
        :param collate_fn: 用于从 dataset 取到的一个 batch 数据进行打包处理的 Callable 函数，其值应该为以下三个: ``[None, "auto", Callable]``.

            * callate_fn 为 ``None`` 时，需要注意的是此时传进来的 datset 类型不能为 :class:`~fastNLP.core.dataset.DataSet` , 当 collate_fn 为 ``None`` 时，
              ``JittorDataLoader`` 调用默认的 Jittor 框架的 ``DataLoader`` 自带的 ``collate_batch`` 作为 callate_fn 的默认值， 其无法处理
              :class:`~fastNLP.core.dataset.DataSet` 的 dataset 对象。
            * callate_fn 为 ``'auto'`` 时，``JittorDataLoader`` 使用 :class:`~fastNLP.core.collators.Collator` 作为 collate_fn 的默认值。
              此时可以配套使用 ``JittorDataLoader`` 的 ``set_pad`` 和 ``set_ignore`` 方法来设置 pad_val 或 忽略某个 field 的检测。
            * collate_fn 为 ``Callable`` 时， 该 Callable 函数应当接受一个 batch 参数作为输入， batch 是一个 List 对象且 List 中的每一条数据都是
              dataset 的一条数据；该 Callable 函数还应当返回一个对象。
            
        """
        # TODO 验证支持replacesampler （以后完成） 增加Sampler
        # 将内部dataset批次设置为1
        if isinstance(dataset, Dataset):
            dataset.set_attrs(batch_size=1, shuffle=False, endless=False)

        # FastNLP Datset, collate_fn not None
        if isinstance(dataset, FDataSet) and collate_fn is None:
            raise ValueError("When use FastNLP DataSet, collate_fn must be not None")

        # 将所有dataset转为jittor类型的dataset
        if not isinstance(dataset, _JittorDataset):
            self.dataset = _JittorDataset(dataset)

        if isinstance(collate_fn, str):
            if collate_fn == "auto":
                if isinstance(self.dataset.dataset, FDataSet):
                    self.collate_fn = deepcopy(self.dataset.dataset.collator)
                    # jittor 比较特殊，只需要保证返回 numpy.array, 其Dataloader会转为jt.var
                    self.collate_fn.set_backend(backend="numpy")
                else:
                    # jittor 比较特殊，只需要保证返回 numpy.array, 其Dataloader会转为jt.var
                    self.collate_fn = Collator(backend="numpy")
            else:
                raise ValueError(f"collate_fn: {collate_fn} must be 'auto'")
        elif isinstance(collate_fn, Callable):
            self.collate_fn = collate_fn
        else:
            self.collate_fn = collate_batch

        self.dataset.set_attrs(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                               num_workers=num_workers, buffer_size=buffer_size, stop_grad=stop_grad,
                               keep_numpy_array=keep_numpy_array, endless=endless)

        self.cur_batch_indices = None

    def __getattr__(self, attr):
        if attr in ["batch_size", "shuffle", "drop_last", "num_workers", "buffer_size", "stop_grad",
                    "keep_numpy_array", "endless", "sampler"]:
            return getattr(self.dataset, attr)
        raise AttributeError(f"{self} has not attribute '{attr}'")

    def __iter__(self):
        # TODO 第一次迭代后不能设置collate_fn，设置是无效的
        if self.cur_batch_indices is None:
            self.dataset.set_attrs(collate_batch=indice_collate_wrapper(self.collate_fn))
        for indices, data in self.dataset.__iter__():
            self.cur_batch_indices = indices
            yield data

    def __len__(self):
        return len(self.dataset)

    def set_pad(self, field_name: Union[str, tuple], pad_val: Union[int, float, None] = 0, dtype=None, backend=None,
                pad_fn: Callable = None) -> Collator:
        """
        如果需要对某个 field 的内容进行特殊的调整，请使用这个函数。

        :param field_name: 需要调整的 field 的名称。如果 :class:`~fastNLP.core.Dataset` 的 :class:`~fastNLP.core.Dataset.__getitem__`
            方法返回的是 dict 类型的，则可以直接使用对应的 field 的 key 来表示，如果是 nested 的 dict，可以使用元组表示多层次的 key，例如
            ``{'a': {'b': 1}}`` 中的使用 ``('a', 'b')`` 如果 ``__getitem__`` 返回的是 Sequence 类型的，则可以使用 *_0*, *_1* 表示序列中
            第 **0** 或 **1** 个元素。如果该 field 在数据中没有找到，则报错；如果 __getitem__ 返回的是就是整体内容，请使用 ``_single`` 。
        :param pad_val: 这个 field 的默认 pad 值。如果设置为 None，则表示该 field 不需要 pad , fastNLP 默认只会对可以 pad 的
            field 进行 pad，所以如果对应 field 本身就不是可以 pad 的形式，可以不需要主动设置为 None 。如果 backend 为 None ，该值
            无意义。
        :param dtype: 对于需要 pad 的 field ，该 field 的数据 dtype 应该是什么。
        :param backend: 可选 ``['raw', 'numpy', 'Jittor', 'paddle', 'jittor', 'auto']`` ，分别代表，输出为 ``list`` , ``numpy.ndarray`` ,
            ``Jittor.Tensor`` , ``paddle.Tensor`` , ``jittor.Var`` 类型。若 ``pad_val`` 为 ``None`` ，该值无意义 。
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

        :param field_name: 需要调整的 field 的名称。如果 :class:`~fastNLP.core.Dataset` 的 :class:`~fastNLP.core.Dataset.__getitem__`
            方法返回的是 dict 类型的，则可以直接使用对应的 field 的 key 来表示，如果是 nested 的 dict，可以使用元组表示多层次的 key，例如
            ``{'a': {'b': 1}}`` 中的使用 ``('a', 'b')`` 如果 ``__getitem__`` 返回的是 Sequence 类型的，则可以使用 *_0*, *_1* 表示序列中
            第 **0** 或 **1** 个元素。
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


def prepare_jittor_dataloader(ds_or_db, batch_size: int = 16, shuffle: bool = False,
                              drop_last: bool = False, num_workers: int = 0, buffer_size: int = 512 * 1024 * 1024,
                              stop_grad: bool = True, keep_numpy_array: bool = False, endless: bool = False,
                              collate_fn: Union[None, str, Callable] = "auto",
                              non_train_batch_size: int = None) \
        -> Union[Dict[str, JittorDataLoader], JittorDataLoader]:
    """
    ``prepare_jittor_dataloader`` 的功能是将输入的单个或多个 dataset 同时转为 :class:`JittorDataLoader` 对象， 详见 :class:`~fastNLP.core.dataloaders.JittorDataLoader`。
    根据 ds_or_db 的类型 ``[DataSet, DataBundle, Dict[name, Dataset]]`` 不同而有不同返回结果, 具体如下:

        * 当 ds_or_db 为 ``DataSet`` 时，``prepare_jittor_dataloader`` 会将使用的除了 non_train_batch_size 和 non_train_sampler 以外的参数来
          帮你实例化一个 :class:`JittorDataLoader` 对象并返回该对象。 详见 :class:`~fastNLP.core.dataloaders.JittorDataLoader`。
        * 当 ds_or_db 为 :class:`~fastNLP.io.DataBundle` 时，``prepare_Jittor_dataloader`` 会遍历 ``DataBundle`` 的数据集的 key-value
          来创建不同的 :class:`JittorDataLoader` 对象；当 key 中包含'train'字符串时，``prepare_jittor_dataloader`` 默认该 value 为 train 数据集，
          会将 batch_size 和 sampler 作为参数，其他 key 不包含 'train' 字符串的数据集则使用 non_train_size 和 non_train_sampler 作为参数。
          最终根据 ``key: JittorDataLoader`` 组成 ``Dict[key, JittorDataLoader]`` 的字典返回。
        * 当 ds_or_db 为 ``Dict[str, DataSet]`` 字典类型时， ``prepare_jittor_dataloader`` 会遍历 该 dict 的的 key-value 来创建不同的
          :class:`JittorDataLoader` 对象；当 key 中包含'train'字符串时，``prepare_Jittor_dataloader`` 默认该 value 为 train 数据集，会将 batch_size 和 sampler 作为参数，
           其他 key 不包含 'train' 字符串的数据集则使用 non_train_size 和 non_train_sampler 作为参数。最终根据  ``key: JittorDataLoader`` 组成
         ``Dict[key, JittorDataLoader]`` 的字典返回。

    :param ds_or_db: 可以有以下三种取值，

        * ds_or_db 为 :class:`~fastNLP.io.DataBundle`, 返回值为 ``Dict[str, TorchDataLoader]`` 的字典
        * ds_or_db 为 ``Dict[str, DataSet]`` 字典， 返回值为 ``Dict[str, TorchDataLoader]`` 的字典
        * ds_or_db 为实现了 __getitem__() 和 __len__() 的对象 ，返回值为:class:`~fastNLP.TorchDataLoader`

    :param non_train_batch_size: 如果传入的 ``ds_or_db`` 为 :class:`Dict` 或 :class:`~fastNLP.io.DataBundle` 对象，可以通过改参数
        设置名称不为 `train` 的其他 ``dataset`` 的 ``batch_size``。 默认为 ``16``。
    :param batch_size: 批次大小，默认为 ``16`` 且当 batch_sampler 为 None 有效。
    :param shuffle: 是否打乱数据集， 默认为 ``False``。
    :param drop_last: 当 ``drop_last=True`` 时，:class:`JittorDataLoader` 会扔掉最后一个长度小于 ``batch_size`` 的 batch 数据;
        若 ``drop_last=False`` , 则会返回该 batch 数据。 默认为 ``False`` 。
    :param num_workers: 当 ``num_workers > 0`` 时, :class:`JittorDataLoader` 会开启 num_workers 个子进程来处理数据， 可以加快
        数据处理速度，但同时也消耗大量内存。 当 ``num_workers=0`` 时， 不开启子进程。 默认为 ``0``。
    :param buffer_size: 每个进程占用的内存空间，默认为512M。主要是配合num_workers使用，用户可以自定义每个进程的内存大小。
    :param stop_grad: 是否不使用梯度， 默认 ``True`` 。
    :param keep_numpy_array: 返回的数据是 ``np.array`` 类型而不是 ``jittor.Var`` 类型，默认为 ``False``
    :param endless: 是否让 :class:`JittorDataLoader` 无限返回数据，也就是将 dataset 循环使用使得返回数据是没有限制的。默认为 ``False``.
    :param collate_fn: 用于从 dataset 取到的一个 batch 数据进行打包处理的 Callable 函数，其值应该为以下三个: ``[None, "auto", Callable]``.

        * callate_fn 为 ``None`` 时，需要注意的是此时传进来的 datset 类型不能为 :class:`~fastNLP.core.dataset.DataSet` , 当 collate_fn 为 ``None`` 时，
          :class:`JittorDataLoader` 调用默认的 Jittor 框架的 ``DataLoader`` 自带的 ``collate_batch`` 作为 callate_fn 的默认值， 其无法处理
          :class:`~fastNLP.core.dataset.DataSet` 的 dataset 对象。
        * callate_fn 为 ``'auto'`` 时，:class:`JittorDataLoader` 使用 :class:`~fastNLP.core.collators.Collator` 作为 collate_fn 的默认值。
          此时可以配套使用 :class:`JittorDataLoader` 的 ``set_pad`` 和 ``set_ignore`` 方法来设置 pad_val 或 忽略某个 field 的检测。
        * collate_fn 为 ``Callable`` 时， 该 Callable 函数应当接受一个 batch 参数作为输入， batch 是一个 List 对象且 List 中的每一条数据都是
          dataset 的一条数据；该 Callable 函数还应当返回一个对象。
    
    :return: 返回数据类型为 :class:`Dict[str, JittorDataLoader]`, :class:`JittorDataLoader` 其中之一，根据输入
        ``ds_or_db`` 变化而变化。
    """
    from fastNLP.io.data_bundle import DataBundle

    if isinstance(ds_or_db, DataBundle):
        dl_bundle = {}
        for name, ds in ds_or_db.iter_datasets():
            if 'train' in name:
                dl_bundle[name] = JittorDataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                                   drop_last=drop_last, num_workers=num_workers,
                                                   buffer_size=buffer_size,
                                                   stop_grad=stop_grad, keep_numpy_array=keep_numpy_array,
                                                   endless=endless,
                                                   collate_fn=collate_fn)
            else:
                dl_bundle[name] = JittorDataLoader(ds,
                                                   batch_size=non_train_batch_size if non_train_batch_size else batch_size,
                                                   shuffle=shuffle,
                                                   drop_last=drop_last, num_workers=num_workers,
                                                   buffer_size=buffer_size,
                                                   stop_grad=stop_grad, keep_numpy_array=keep_numpy_array,
                                                   endless=endless,
                                                   collate_fn=collate_fn)
        return dl_bundle

    elif isinstance(ds_or_db, Dict):
        ds_dict = {}
        for name, ds in ds_or_db.items():
            if 'train' in name:
                dl = JittorDataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                      drop_last=drop_last, num_workers=num_workers, buffer_size=buffer_size,
                                      stop_grad=stop_grad, keep_numpy_array=keep_numpy_array, endless=endless,
                                      collate_fn=collate_fn)
            else:
                dl = JittorDataLoader(ds,
                                      batch_size=non_train_batch_size if non_train_batch_size else batch_size,
                                      shuffle=shuffle,
                                      drop_last=drop_last, num_workers=num_workers,
                                      buffer_size=buffer_size,
                                      stop_grad=stop_grad, keep_numpy_array=keep_numpy_array,
                                      endless=endless,
                                      collate_fn=collate_fn)
            ds_dict[name] = dl
        return ds_dict

    elif isinstance(ds_or_db, HasLenGetitemType):
        dl = JittorDataLoader(ds_or_db, batch_size=batch_size, shuffle=shuffle,
                              drop_last=drop_last, num_workers=num_workers, buffer_size=buffer_size,
                              stop_grad=stop_grad, keep_numpy_array=keep_numpy_array, endless=endless,
                              collate_fn=collate_fn)
        return dl

    else:
        raise ValueError(f"ds_or_db: {ds_or_db} must be fastnlp dataset or data_bundle or mapping!")
