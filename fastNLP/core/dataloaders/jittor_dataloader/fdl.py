__all__ = [
    'JittorDataLoader',
    'prepare_jittor_dataloader'
]

from typing import Callable, Optional, List, Union
from copy import deepcopy

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    from jittor.dataset.utils import collate_batch
    from jittor.dataset import Dataset
else:
    from fastNLP.core.dataset import DataSet as Dataset

from fastNLP.core.collators import Collator
from fastNLP.core.dataloaders.utils import indice_collate_wrapper
from fastNLP.core.dataset import DataSet as FDataSet


class _JittorDataset(Dataset):
    """
    对用户传的dataset进行封装，以便JittorDataLoader能够支持使用自定义的dataset
    """

    def __init__(self, dataset) -> None:
        super(_JittorDataset, self).__init__()
        self.dataset = dataset
        self.total_len = len(dataset)

    def __getitem__(self, item):
        return (item, self.dataset[item])


class JittorDataLoader:
    """
    提供给使用jittor框架的DataLoader函数，其能够自动检测数据的类型并判断是否能够pad，若能会自动pad数据，默认pad_val=0;
    用户可以调用set_pad方法来更改pad_val的值，也可以自定义针对某个field的callate_fn传入到set_field；若用户不想自动pad某个field,
    则可以调用set_ignore来忽略对某个field的检测和pad。值得注意的是JittorDataLoader输入dataset只要是实现了__getitem__和__len__方法即可。

    """

    def __init__(self, dataset, batch_size: int = 16, shuffle: bool = True,
                 drop_last: bool = False, num_workers: int = 0, buffer_size: int = 512 * 1024 * 1024,
                 stop_grad: bool = True, keep_numpy_array: bool = False, endless: bool = False,
                 collate_fn: Union[None, str, Callable] = "auto") -> None:
        """

        :param dataset: 实现``__getitem__``和``__len__``的dataset
        :param batch_size: 批次大小
        :param shuffle: 是否打乱数据集
        :param drop_last: 是否去掉最后一个不符合``batch_size``的数据
        :param num_workers: 进程的数量，当``num_workers=0``时不开启多进程
        :param buffer_size: 每个进程占用的内存空间，默认为512M。主要是配合num_workers使用，用户可以自定义每个进程的内存大小。
        :param stop_grad:
        :param keep_numpy_array: 返回的数据是``np.array`类`型而不是``jittor.array``类型，默认为``False``
        :param endless: 是否让``JittorDataLoader``无限返回数据，也就是将dataset循环使用使得返回数据是没有限制的。默认为``False``.
        :param collate_fn: 用来对从dataset取到的数据进行打包处理成batch的callable函数，其值应该为一下三个:``[None, "auto", callable]``.

            * ``callate_fn=None``时，第一点值得注意的是此时传进来的datset不能为``fastNLP``的dataset,采用fastNLP的dataset时，``collate_fn``不能为``None``;
            第二点注意的是此时``JittorDataLoader``会调用默认的`callate_batch`函数对sampler到的数据进行简单打包，组成一个batch返回。`
            * ``callate_fn="auto"``时，``JittorDataLoader``会自动调用``fastNLP``自带的``Collator``，其会自动检测dataset的每个``field``,
            并判断是否能够pad处理，若能则会自动进行pad操作，默认``pad_val=0``。若想要更改其值，可调用``set_pad``方法;若不想自动pad某个field，
            可以调用``set_ignore``方法忽略某个field。
            * ``callate_fn=callable``时，callable函数是用户自定义的callate_fn函数，此时``JittorDataLoader``会调用传进来的callable函数对
            数据进行打包处理并返回。值得注意的是用户自定义的callable函数的输入为batch,batch为list类型数据，其中batch的每一条数据都为dataset的一条数据。

        """
        # TODO 验证支持replacesampler （以后完成） 增加Sampler
        # 将内部dataset批次设置为1
        if isinstance(dataset, Dataset):
            dataset.set_attrs(batch_size=1)

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


def prepare_jittor_dataloader():
    ...
