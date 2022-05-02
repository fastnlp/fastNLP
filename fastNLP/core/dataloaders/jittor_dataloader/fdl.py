__all__ = [
    'JittorDataLoader',
    'prepare_jittor_dataloader'
]

from typing import Callable, Optional, List, Union

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    from jittor.dataset.utils import collate_batch
    from jittor.dataset import Dataset
else:
    from fastNLP.core.dataset import DataSet as Dataset
from fastNLP.core.utils.jittor_utils import jittor_collate_wraps
from fastNLP.core.collators import Collator
from fastNLP.core.utils.utils import indice_collate_wrapper
from fastNLP.core.dataset import DataSet as FDataSet


class _JittorDataset(Dataset):
    """
    对用户传的dataset进行封装，以便JittorDataLoader能够支持使用自定义的dataset使用jittor的dataset
    """

    def __init__(self, dataset) -> None:
        super(_JittorDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, item):
        return (item, self.dataset[item])

    def __len__(self) -> int:
        return len(self.dataset)

    # def __getattr__(self, item):
    #     # jittor的Dataset没有的方法而用户的dataset存在且实现了getattribute方法，此时用户可以调用
    #     try:
    #         self.dataset.__getattribute__(item)
    #     except Exception as e:
    #         raise e


class JittorDataLoader:
    """
    提供给使用jittor框架的DataLoader函数，提供了auto_collate的功能， 支持实现了__getitem__和__len__的dataset
    """

    def __init__(self, dataset, batch_size: int = 16, shuffle: bool = False,
                 drop_last: bool = False, num_workers: int = 0, buffer_size: int = 512 * 1024 * 1024,
                 stop_grad: bool = True, keep_numpy_array: bool = False, endless: bool = False,
                 collate_fn: Union[None, str, Callable] = "auto") -> None:
        """

        :param dataset: 实现__getitem__和__len__的dataset
        :param batch_size: 批次大小
        :param shuffle: 是否打乱数据集
        :param drop_last: 是否去掉最后一个不符合batch_size的数据
        :param num_workers: 进程的数量，当num_workers=0时不开启多进程
        :param buffer_size:
        :param stop_grad:
        :param keep_numpy_array:
        :param endless:
        :param collate_fn: 对取得到的数据进行打包的callable函数
        :param as_numpy: 返回数据是否设置为numpy类型，否则为torch.tensor类型
        """
        # TODO 支持fastnlp dataset
        # TODO 验证支持replacesampler （以后完成）
        # 是否为 jittor 类型的 dataset
        if isinstance(collate_fn, str):
            if collate_fn == "auto":
                if isinstance(dataset, FDataSet):
                    self._collate_fn = dataset.collator
                    self._collate_fn.set_backend(backend="jittor")
                else:
                    self._collate_fn = Collator(backend="jittor")
            else:
                raise ValueError(f"collate_fn: {collate_fn} must be 'auto'")
        elif isinstance(collate_fn, Callable):
            if collate_fn is not collate_batch:
                self._collate_fn = collate_fn
        else:
            self._collate_fn = collate_batch

        self.dataset = _JittorDataset(dataset)

        self.dataset.set_attrs(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                               num_workers=num_workers, buffer_size=buffer_size, stop_grad=stop_grad,
                               keep_numpy_array=keep_numpy_array, endless=endless)
        if isinstance(self.dataset.dataset, Dataset):
            self.dataset.dataset.set_attrs(batch_size=1)
        # 用户提供了 collate_fn，则会自动代替 jittor 提供 collate_batch 函数
        # self._collate_fn = _collate_fn

    def __iter__(self):
        # TODO 第一次迭代后不能设置collate_fn，设置是无效的
        self.collate_fn = self._collate_fn
        if self.cur_batch_indices is None:
            self.dataset.set_attrs(collate_batch=indice_collate_wrapper(self.collate_fn))
        for indices, data in self.dataset.__iter__():
            self.cur_batch_indices = indices
            yield data

    def __len__(self):
        if self.dataset.drop_last:
            return len(self.dataset) // self.dataset.batch_size
        return (len(self.dataset) - 1) // self.dataset.batch_size + 1

    def set_pad(self, field_name: Union[str, tuple], pad_val: Union[int, float, None] = 0, dtype=None, backend=None,
                pad_fn: Callable = None) -> "JittorDataLoader":
        """
            如果需要对某个 field 的内容进行特殊的调整，请使用这个函数。

            :param field_name: 需要调整的 field 的名称。如果 Dataset 的 __getitem__ 方法返回的是 dict 类型的，则可以直接使用对应的
                field 的 key 来表示，如果是 nested 的 dict，可以使用元组表示多层次的 key，例如 {'a': {'b': 1}} 中的使用 ('a', 'b');
                如果 __getitem__ 返回的是 Sequence 类型的，则可以使用 '_0', '_1' 表示序列中第 0 或 1 个元素。如果该 field 在数据中没
                有找到，则报错；如果 __getitem__ 返回的是就是整体内容，请使用 "_single" 。
            :param pad_val: 这个 field 的默认 pad 值。如果设置为 None，则表示该 field 不需要 pad , fastNLP 默认只会对可以 pad 的
                field 进行 pad，所以如果对应 field 本身就不是可以 pad 的形式，可以不需要主动设置为 None 。
            :param dtype: 对于需要 pad 的 field ，该 field 的数据 dtype 应该是什么。
            :param backend: 可选[None, 'numpy', 'torch', 'paddle', 'jittor']，分别代表，输出为 list, numpy.ndarray, torch.Tensor,
                paddle.Tensor, jittor.Var 类型。若 pad_val 为 None ，该值只能为 None 或 numpy 。
            :param pad_fn: 指定当前 field 的 pad 函数，传入该函数则 pad_val, dtype, backend 等参数失效。pad_fn 的输入为当前 field 的
                batch 形式。 Collator 将自动 unbatch 数据，然后将各个 field 组成各自的 batch 。pad_func 的输入即为 field 的 batch
                形式，输出将被直接作为结果输出。
            :return: 返回 Collator 自身
        """
        if isinstance(self._collate_fn, Collator):
            self._collate_fn.set_pad(field_name=field_name, pad_val=pad_val, dtype=dtype, pad_fn=pad_fn,
                                     backend=backend)
            return self
        else:
            raise ValueError(f"collate_fn is not fastnlp collator")

    def set_ignore(self, *field_names) -> "JittorDataLoader":
        """
        如果有的内容不希望输出，可以在此处进行设置，被设置的 field 将在 batch 的输出中被忽略。
        Ex::
            collator.set_ignore('field1', 'field2')

        :param field_names: 需要忽略的 field 的名称。如果 Dataset 的 __getitem__ 方法返回的是 dict 类型的，则可以直接使用对应的
            field 的 key 来表示，如果是 nested 的 dict，可以使用元组来表示，例如 {'a': {'b': 1}} 中的使用 ('a', 'b'); 如果
            __getitem__ 返回的是 Sequence 类型的，则可以使用 '_0', '_1' 表示序列中第 0 或 1 个元素。
        :return: 返回 Collator 自身
        """
        if isinstance(self._collate_fn, Collator):
            self._collate_fn.set_ignore(*field_names)
            return self
        else:
            raise ValueError(f"collate_fn is not fastnlp collator")

    def get_batch_indices(self) -> List[int]:
        """
        获取当前数据的idx

        :return:
        """
        return self.cur_batch_indices


def prepare_jittor_dataloader():
    ...
