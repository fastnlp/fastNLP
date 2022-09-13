r"""
:class:`~fastNLP.core.dataset.DataSet` 是 fastNLP 中用于承载数据的容器。可以将 DataSet 看做是一个表格，
每一行是一个 sample (在 fastNLP 中被称为 :mod:`~fastNLP.core.dataset.instance` )，
每一列是一个 feature (在 fastNLP 中称为 :mod:`~fastNLP.core.dataset.field` )。

.. csv-table:: Following is a demo layout of DataSet
   :header: "sentence", "words", "seq_len"

   "This is the first instance .", "[This, is, the, first, instance, .]", 6
   "Second instance .", "[Second, instance, .]", 3
   "Third instance .", "[Third, instance, .]", 3
   "...", "[...]", "..."

在 fastNLP 内部每一行是一个 :class:`~fastNLP.core.dataset.Instance` 对象； 每一列是一个 :class:`~fastNLP.core.dataset.FieldArray` 对象。

----------------------------
1.DataSet的创建
----------------------------

创建DataSet主要有以下的3种方式

1.1 传入dict
----------------------------

    .. code-block::

        from fastNLP import DataSet
        data = {'sentence':["This is the first instance .", "Second instance .", "Third instance ."],
                'words': [['this', 'is', 'the', 'first', 'instance', '.'], ['Second', 'instance', '.'], ['Third', 'instance', '.'],
                'seq_len': [6, 3, 3]}
        dataset = DataSet(data)
        # 传入的 dict 的每个 key 的 value 应该为具有相同长度的l ist

1.2 通过 Instance 构建
----------------------------

    .. code-block::

        from fastNLP import DataSet
        from fastNLP import Instance
        dataset = DataSet()
        instance = Instance(sentence="This is the first instance",
                            words=['this', 'is', 'the', 'first', 'instance', '.'],
                            seq_len=6)
        dataset.append(instance)
        # 可以继续 append 更多内容，但是 append 的 instance 应该和第一个 instance 拥有完全相同的 field

1.3 通过 List[Instance] 构建
--------------------------------------

    .. code-block::

        from fastNLP import DataSet
        from fastNLP import Instance
        instances = []
        winstances.append(Instance(sentence="This is the first instance",
                            ords=['this', 'is', 'the', 'first', 'instance', '.'],
                            seq_len=6))
        instances.append(Instance(sentence="Second instance .",
                            words=['Second', 'instance', '.'],
                            seq_len=3))
        dataset = DataSet(instances)

--------------------------------------
2.DataSet 与预处理
--------------------------------------

常见的预处理有如下几种：

2.1 从某个文本文件读取内容
--------------------------------------

    .. code-block::

        from fastNLP import DataSet
        from fastNLP import Instance
        dataset = DataSet()
        filepath = 'some/text/file'
        # 假设文件中每行内容如下(sentence  label):
        #    This is a fantastic day    positive
        #    The bad weather    negative
        #    .....
        with open(filepath, 'r') as f:
            for line in f:
                sent, label = line.strip().split('\t')
                dataset.append(Instance(sentence=sent, label=label))


2.2 对 DataSet 中的内容处理
--------------------------------------

    .. code-block::

        from fastNLP import DataSet
        data = {'sentence':["This is the first instance .", "Second instance .", "Third instance ."]}
        dataset = DataSet(data)
        # 将句子分成单词形式, 详见DataSet.apply()方法, 可以开启多进程来加快处理， 也可以更改展示的bar，目前支持 ``['rich', 'tqdm', None]``,
        # 详细内容可以见 :class:`~fastNLP.core.dataset.DataSet`, 需要注意的时匿名函数不支持多进程
        dataset.apply(lambda ins: ins['sentence'].split(), new_field_name='words',
            progress_des='Main',progress_bar='rich')
        # 或使用DataSet.apply_field()
        dataset.apply_field(lambda sent:sent.split(), field_name='sentence', new_field_name='words',
            progress_des='Main',progress_bar='rich')
        # 除了匿名函数，也可以定义函数传递进去
        def get_words(instance):
            sentence = instance['sentence']
            words = sentence.split()
            return words
        dataset.apply(get_words, new_field_name='words'， num_proc=2, progress_des='Main',progress_bar='rich')

2.3 删除DataSet的内容
--------------------------------------

    .. code-block::

        from fastNLP import DataSet
        dataset = DataSet({'a': list(range(-5, 5))})
        # 返回满足条件的 instance,并放入 DataSet 中
        dropped_dataset = dataset.drop(lambda ins:ins['a']<0, inplace=False)
        # 在 dataset 中删除满足条件的i nstance
        dataset.drop(lambda ins:ins['a']<0)  # dataset 的 instance数量减少
        #  删除第 3 个 instance
        dataset.delete_instance(2)
        #  删除名为 'a' 的 field
        dataset.delete_field('a')


2.4 遍历DataSet的内容
--------------------------------------

    .. code-block::

        for instance in dataset:
            # do something

2.5 一些其它操作
--------------------------------------

    .. code-block::

        #  检查是否存在名为 'a' 的 field
        dataset.has_field('a')  # 或 ('a' in dataset)
        #  将名为 'a' 的 field 改名为 'b'
        dataset.rename_field('a', 'b')
        #  DataSet 的长度
        len(dataset)

"""

__all__ = [
    "DataSet",
    "ApplyResultException"
]

import _pickle as pickle
from copy import deepcopy
from typing import Optional, List, Callable, Union, Dict, Any, Mapping
from types import LambdaType
import sys
import time

import numpy as np

from .field import FieldArray
from .instance import Instance
from fastNLP.core.utils.utils import pretty_table_printer, deprecated
from fastNLP.core.collators import Collator
from fastNLP.core.utils.rich_progress import f_rich_progress, DummyFRichProgress
from fastNLP.core.utils.tqdm_progress import f_tqdm_progress
from ..log import logger
from fastNLP.core.utils.dummy_class import DummyClass
from ..utils.utils import _get_fun_msg


progress_bars = {
    'rich': f_rich_progress,
    'tqdm': f_tqdm_progress
}


class ApplyResultException(Exception):
    def __init__(self, msg, index=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index  # 标示在哪个数据遭遇到问题了


def _apply_single(ds=None, _apply_field=None, func: Optional[Callable] = None, progress_bar: str = 'rich',
                  desc: str = None) -> list:
    """
    对数据集进行处理封装函数，以便多进程使用

    :param ds: 实现了 __getitem__() 和 __len__() 的对象
    :param _apply_field: 需要处理数据集的 field_name
    :param func: 用户自定义的 func
    :param desc: 进度条的描述字符
    :param progress_bar: 显示 progress_bar 的方式，支持 `["rich", "tqdm", None]`。
    :return:
    """
    progress_bar = progress_bars.get(progress_bar, DummyFRichProgress())
    desc = desc if desc else "Processing"
    task_id = progress_bar.add_task(description=desc, total=len(ds))
    results = []
    idx = -1

    try:
        for idx, ins in enumerate(ds):
            if _apply_field is not None:
                results.append(func(ins[_apply_field]))
            else:
                results.append(func(ins))
            progress_bar.update(task_id, advance=1)

    except BaseException as e:
        if idx != -1:
            logger.error("Exception happens at the `{}`th instance.".format(idx))
        raise e
    finally:
        progress_bar.destroy_task(task_id)
    return results


def _multi_proc(ds, _apply_field, func, counter, queue):
    """
    对数据集进行处理封装函数，以便多进程使用

    :param ds: 实现了 __getitem__() 和 __len__() 的对象
    :param _apply_field: 需要处理数据集的 field_name
    :param func: 用户自定义的 func
    :param counter: 计数器
    :param queue: 多进程时，将结果输入到这个 queue 中
    :return:
    """
    idx = -1
    import contextlib
    null = DummyClass()
    with contextlib.redirect_stdout(null):  # 避免打印触发 rich 的锁
        logger.set_stdout(stdout='raw')
        results = []
        try:
            for idx, ins in enumerate(ds):
                if _apply_field is not None:
                    res = func(ins[_apply_field])
                else:
                    res = func(ins)
                results.append(res)
                with counter.get_lock():
                    counter.value += 1
        except BaseException as e:
            if idx != -1:
                logger.error("Exception happens at the `{}`th instance.".format(idx))
            raise e
    queue.put(pickle.dumps(results))


class DataSet:
    r"""
    fastNLP的数据容器。

    Example::

        from fastNLP.core.dataset import DataSet, Instance
        data = {'x': [[1, 0, 1], [0, 1, 1], 'y': [0, 1]}
        data1 = [Instance(x=[1,0,1],y=0), Instance(x=[0,1,1],y=1)]
        ds = DataSet(data)
        ds = DataSet(data1)

    fastNLP的 DataSet 是 key-value 存储形式， 目前支持两种初始化方式，输入 data 分别为 ``List[:class:`~fastNLP.core.dataset.Instance`]`` 和
    ``Dict[str, List[Any]]``。

        * 当 data 为 ``List[:class:`~fastNLP.core.dataset.Instance`]`` 时,  每个 ``Instance`` 的 field_name 需要保持一致。
          Instance 详见 :class:`~fastNLP.core.dataset.Instance` 。
        * 当 data 为 ``Dict[str, List[Any]]`` 时， 则每个 key 的 value 应该为等长的 list， 否则不同 field 的长度不一致。

    :param data: 初始化的内容，其只能为两种类型，分别为 ``List[:class:`~fastNLP.core.dataset.Instance`]`` 和
        ``Dict[str, List[Any]]``。

        * 当 data 为 ``List[:class:`~fastNLP.core.dataset.Instance`]`` 时,  每个 ``Instance`` 的 field_name 需要保持一致。
          Instance 详见 :class:`~fastNLP.core.dataset.Instance` 。
        * 当 data 为 ``Dict[str, List[Any]] 时， 则每个 key 的 value 应该为等长的 list， 否则不同 field 的长度不一致。
    """
    def __init__(self, data: Union[List[Instance], Dict[str, List[Any]], None] = None):
        self.field_arrays = {}
        self._collator = Collator()
        if data is not None:
            if isinstance(data, Dict):
                length_set = {}
                for key, value in data.items():
                    length_set[key] = len(value)
                assert len(set(length_set.values())) == 1, f"Fields must all be of same length, instead of {length_set}."
                for key, value in data.items():
                    self.add_field(field_name=key, fields=value)
            elif isinstance(data, List):
                for ins in data:
                    assert isinstance(ins, Instance), "Must be Instance type, not {}.".format(type(ins))
                    self.append(ins)
            else:
                raise ValueError("data only be dict or list type.")

    def __contains__(self, item):
        return item in self.field_arrays

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def _inner_iter(self):
        class Iter_ptr:
            def __init__(self, dataset, idx):
                self.dataset = dataset
                self.idx = idx

            def __getitem__(self, item):
                assert item in self.dataset.field_arrays, "no such field:{} in Instance {}".format(item, self.dataset[
                    self.idx])
                assert self.idx < len(self.dataset.field_arrays[item]), "index:{} out of range".format(self.idx)
                return self.dataset.field_arrays[item][self.idx]

            def __setitem__(self, key, value):
                raise TypeError("You cannot modify value directly.")

            def items(self):
                ins = self.dataset[self.idx]
                return ins.items()

            def __repr__(self):
                return self.dataset[self.idx].__repr__()

        def inner_iter_func():
            for idx in range(len(self)):
                yield Iter_ptr(self, idx)

        return inner_iter_func()

    def __getitem__(self, idx: Union[int, slice, str, list]):
        r"""
        去 DataSet 的内容， 根据 idx 类型不同有不同的返回值。 包括四种类型 ``[int, slice, str, list]``

            * 当 idx 为 ``int`` 时， idx 的值不能超过 ``DataSet`` 的长度, 会返回一个 ``Instance``, 详见
            :class:`~fastNLP.core.dataset.Instance`
            * 当 idx 为 ``slice`` 时， 会根据 slice 的内容创建一个新的 DataSet，其包含 slice 所有内容并返回。
            * 当 idx 为 ``str`` 时， 该 idx 为 DataSet 的 field_name, 其会返回该 field_name 的所有内容， 为 list 类型。
            * 当 idx 为 ``list`` 时， 该 idx 的 list 内全为 int 数字， 其会取出所有内容组成一个新的 DataSet 返回。

        Example::

            from fastNLP.core.dataset import DataSet

            ds = DataSet({'x': [[1, 0, 1], [0, 1, 1] * 100, 'y': [0, 1] * 100})
            ins = ds[0]
            sub_ds = ds[0:100]
            sub_ds= ds[[1, 0, 3, 2, 1, 4]]
            field = ds['x']

        :param idx: 用户传入参数
        :return:
        """
        if isinstance(idx, int):
            return Instance(**{name: self.field_arrays[name][idx] for name in self.field_arrays})
        elif isinstance(idx, slice):
            if idx.start is not None and (idx.start >= len(self) or idx.start <= -len(self)):
                raise RuntimeError(f"Start index {idx.start} out of range 0-{len(self) - 1}")
            dataset = DataSet()
            for field_name, field in self.field_arrays.items():
                dataset.add_field(field_name=field_name, fields=field.content[idx])
            dataset._collator = deepcopy(self.collator)
            return dataset
        elif isinstance(idx, str):
            if idx not in self:
                raise KeyError("No such field called {} in DataSet.".format(idx))
            return self.field_arrays[idx]
        elif isinstance(idx, list):
            dataset = DataSet()
            for i in idx:
                assert isinstance(i, int), "Only int index allowed."
                instance = self[i]
                dataset.append(instance)
            dataset._collator = deepcopy(self.collator)
            return dataset
        else:
            raise KeyError("Unrecognized type {} for idx in __getitem__ method".format(type(idx)))

    def __setitem__(self, key, value):
        assert isinstance(key, int) and key<len(self)
        assert isinstance(value, Instance) or isinstance(value, Mapping)
        ins_keys = set(value.keys())
        ds_keys = set(self.get_field_names())

        if len(ins_keys - ds_keys) != 0:
            raise KeyError(f"The following keys are not found in the Dataset:{list(ins_keys - ds_keys)}.")
        if len(ds_keys - ins_keys) != 0:
            raise KeyError(f"The following keys are not found in the Instance:{list(ds_keys - ins_keys)}.")

        for field_name, field in self.field_arrays.items():
            field[key] = value[field_name]

    def __getattribute__(self, item):
        return object.__getattribute__(self, item)

    def __getattr__(self, item):
        # Not tested. Don't use !!
        if isinstance(item, str) and item in self.field_arrays:
            return self.field_arrays[item]
        else:
            raise AttributeError(f"Dataset has no attribute named:{item}.")

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__

    def __len__(self):
        r"""
        获取 DataSet 的长度

        :return
        """
        if len(self.field_arrays) == 0:
            return 0
        field = iter(self.field_arrays.values()).__next__()
        return len(field)

    def __repr__(self):
        return str(pretty_table_printer(self))

    def append(self, instance: Instance) -> None:
        r"""
        将一个 ``instance`` 对象 append 到 DataSet 后面。详见 :class:`~fastNLP.core.dataset.Instance`

        :param instance: 若 DataSet 不为空，则 instance 应该拥有和 DataSet 完全一样的 field；
        """
        if len(self.field_arrays) == 0:
            # DataSet has no field yet
            for name, field in instance.items():
                # field = field.tolist() if isinstance(field, np.ndarray) else field
                self.field_arrays[name] = FieldArray(name, [field])  # 第一个样本，必须用list包装起来
        else:
            if len(self.field_arrays) != len(instance.fields):
                raise ValueError(
                    "DataSet object has {} fields, but attempt to append an Instance object with {} fields."
                        .format(len(self.field_arrays), len(instance.fields)))
            for name, field in instance.items():
                assert name in self.field_arrays, f'Field:`{name}` is not found in {self.field_arrays.keys()}'
                try:
                    self.field_arrays[name].append(field)
                except Exception as e:
                    logger.error(f"Cannot append to field:{name}.")
                    raise e

    def add_fieldarray(self, field_name: str, fieldarray: FieldArray) -> None:
        r"""
        将 ``fieldarray`` 添加到 DataSet 中.

        :param field_name: 新加入的 field 的名称；
        :param fieldarray: 需要加入 DataSet 的 field 的内容, 详见 :class:`~fastNLP.core.dataset.FieldArray` ；
        :return:
        """
        if not isinstance(fieldarray, FieldArray):
            raise TypeError("Only fastNLP.FieldArray supported.")
        if len(self) != len(fieldarray):
            raise RuntimeError(f"The field to add must have the same size as dataset. "
                               f"Dataset size {len(self)} != field size {len(fieldarray)}")
        fieldarray.name = field_name
        self.field_arrays[field_name] = fieldarray

    def add_field(self, field_name: str, fields: list) -> None:
        r"""
        新增一个 field， 需要注意的是 fields 的长度跟 DataSet 长度一致

        :param field_name: 新增的 field 的名称；
        :param fields: 需要新增的 field 的内容；
        """

        if len(self.field_arrays) != 0:
            if len(self) != len(fields):
                raise RuntimeError(f"The field to add must have the same size as dataset. "
                                   f"Dataset size {len(self)} != field size {len(fields)}")
        self.field_arrays[field_name] = FieldArray(field_name, fields)

    def delete_instance(self, index: int):
        r"""
        删除第 ``index`` 个 Instance

        :param index: 需要删除的 instance 的 index，序号从 `0` 开始。
        """
        assert isinstance(index, int), "Only integer supported."
        if len(self) <= index:
            raise IndexError("{} is too large for as DataSet with {} instances.".format(index, len(self)))
        if len(self) == 1:
            self.field_arrays.clear()
        else:
            for field in self.field_arrays.values():
                field.pop(index)
        return self

    def delete_field(self, field_name: str):
        r"""
        删除名为 ``field_name`` 的 field

        :param field_name: 需要删除的 field 的名称；
        """
        if self.has_field(field_name):
            self.field_arrays.pop(field_name)
        else:
            raise KeyError(f"Field:{field_name} not found in DataSet.")
        return self

    def copy_field(self, field_name: str, new_field_name: str):
        r"""
        深度 copy 名为 ``field_name`` 的 field 到 ``new_field_name``

        :param field_name: 需要 copy 的 field；
        :param new_field_name: copy 生成的 field 名称；
        :return: 数据集自身；
        """
        if not self.has_field(field_name):
            raise KeyError(f"Field:{field_name} not found in DataSet.")
        fieldarray = deepcopy(self.get_field(field_name))
        fieldarray.name = new_field_name
        self.add_fieldarray(field_name=new_field_name, fieldarray=fieldarray)
        return self

    def has_field(self, field_name: str) -> bool:
        r"""
        判断 DataSet 中是否有名为 ``field_name`` 这个 field

        :param field_name: field 的名称；
        :return: 表示是否有名为 ``field_name`` 这个 field；
        """
        if isinstance(field_name, str):
            return field_name in self.field_arrays
        return False

    def get_field(self, field_name: str) -> FieldArray:
        r"""
        获取名为 ``field_name`` 的 field

        :param field_name: field 的名称；
        :return: 一个 :class:`~fastNLP.core.dataset.FieldArray` 对象；
        """
        if field_name not in self.field_arrays:
            raise KeyError("Field name {} not found in DataSet".format(field_name))
        return self.field_arrays[field_name]

    def get_all_fields(self) -> dict:
        r"""
        :return: 一个 dict，key 为 field_name, value为对应的 :class:`~fastNLP.core.dataset.FieldArray` 对象。
        """
        return self.field_arrays

    def get_field_names(self) -> list:
        r"""
        :return: 一个 list，包含所有 field 的名字
        """
        return sorted(self.field_arrays.keys())

    def get_length(self) -> int:
        r"""
        获取 DataSet 的元素数量

        :return: DataSet 中 Instance 的个数。
        """
        return len(self)

    def rename_field(self, field_name: str, new_field_name: str):
        r"""
        将某个 field 重新命名.

        :param field_name: 原来的 field 名称；
        :param new_field_name: 修改为 new_name；
        """
        if field_name in self.field_arrays:
            self.field_arrays[new_field_name] = self.field_arrays.pop(field_name)
            self.field_arrays[new_field_name].name = new_field_name
        else:
            raise KeyError("DataSet has no field named {}.".format(field_name))
        return self

    def apply_field(self, func: Callable, field_name: str = None,
                    new_field_name: str = None, num_proc: int = 0,
                    progress_desc: str = None, progress_bar: str = 'rich'):
        r"""
        将 :class:`DataSet` 每个 ``instance`` 中为 ``field_name`` 的 field 传给函数 ``func``，并写入到 ``new_field_name``
        中。

        :param func: 对指定 field 进行处理的函数，注意其输入应为 ``instance`` 中名为 ``field_name`` 的 field 的内容，返回值将被
            写入至 ``new_field_name`` 中。
        :param field_name: 传入 ``func`` 的 field 名称；
        :param new_field_name: 函数执行结果写入的 ``field`` 名称。该函数会将 ``func`` 返回的内容放入到 ``new_field_name`` 对
            应的 ``field`` 中，注意如果名称与已有的 field 相同则会进行覆盖。如果为 ``None`` 则不会覆盖和创建 field ；
        :param num_proc: 使用进程的数量。
        
            .. note::
            
                由于 ``python`` 语言的特性，设置该参数后会导致相应倍数的内存增长，这可能会对您程序的执行带来一定的影响。另外，使用多进程时，
                ``func`` 函数中的打印将不会输出。

        :param progress_desc: 如果不为 ``None``，则会显示当前正在处理的进度条的名称；
        :param progress_bar: 显示进度条的方式，支持 ``["rich", "tqdm", None]``。
        :return: 从函数 ``func`` 中得到的返回值；
        """
        assert len(self) != 0, "Null DataSet cannot use apply_field()."
        if not self.has_field(field_name=field_name):
            raise KeyError("DataSet has no field named `{}`.".format(field_name))

        try:
            results = self._apply_process(num_proc=num_proc, func=func, progress_bar=progress_bar,
                                          progress_desc=progress_desc, _apply_field=field_name)
        except BaseException as e:
            raise e

        if new_field_name is not None:
            self.add_field(field_name=new_field_name, fields=results)
        return results

    def apply_field_more(self, func: Callable = None, field_name: str = None,
                         modify_fields: bool = True, num_proc: int = 0,
                         progress_desc: str = None, progress_bar: str = 'rich'):
        r"""
        将 ``DataSet`` 中的每个 ``Instance`` 中的名为 `field_name` 的 field 传给 ``func``，并获取它的返回值。
        ``func`` 可以返回一个或多个 field 上的结果。

        .. note::
            ``apply_field_more`` 与 ``apply_field`` 的区别参考 :meth:`~fastNLP.core.dataset.DataSet.apply_more` 中关于 ``apply_more`` 与
            ``apply`` 区别的介绍。

        :param func: 对指定 field 进行处理的函数，注意其输入应为 ``instance`` 中名为 ``field_name`` 的 field 的内容；返回值为一个字典，
            key 是field 的名字，value 是对应的结果
        :param field_name: 传入 ``func`` 的 field 名称；
        :param modify_fields: 是否用结果修改 ``DataSet`` 中的 ``Field`` ， 默认为 ``True``
        :param num_proc: 使用进程的数量。
        
            .. note::
            
                由于 ``python`` 语言的特性，设置该参数后会导致相应倍数的内存增长，这可能会对您程序的执行带来一定的影响。另外，使用多进程时，
                ``func`` 函数中的打印将不会输出。

        :param progress_desc: 如果不为 ``None``，则会显示当前正在处理的进度条的名称；
        :param progress_bar: 显示进度条的方式，支持 ``["rich", "tqdm", None]``。
        :return: 一个字典
        """
        assert len(self) != 0, "Null DataSet cannot use apply_field()."
        if not self.has_field(field_name=field_name):
            raise KeyError("DataSet has no field named `{}`.".format(field_name))
        idx = -1
        results = {}
        apply_out = self._apply_process(num_proc, func, progress_desc=progress_desc,
                                        progress_bar=progress_bar, _apply_field=field_name)
        #   只检测第一个数据是否为dict类型，若是则默认所有返回值为dict；否则报错。
        if not isinstance(apply_out[0], Mapping):
            raise Exception(f"The result of func is not a Mapping, but a {type(apply_out[0])}")

        for key, value in apply_out[0].items():
            results[key] = [value]
        #   尝试合并所有dict数据, idx+1 的原因是第一条数据不可能出现错误，默认第一条数据为准
        try:
            for idx, per_out in enumerate(apply_out[1:]):
                if len(set(results.keys()) - set(per_out.keys())):
                    raise ApplyResultException("apply results have different fields", idx + 1)
                for key, value in per_out.items():
                    results[key].append(value)

        except Exception as e:
            if idx != -1:
                logger.error("Exception happens at the `{}`th instance.".format(idx + 1))
            raise e

        if modify_fields is True:
            for field, result in results.items():
                self.add_field(field_name=field, fields=result)

        return results

    def _apply_process(self, num_proc: int = 0, func: Callable = None,
                       progress_bar: str = 'rich', _apply_field: str = None,
                       progress_desc: str = 'Main') -> list:
        """
        :param num_proc: 使用进程的数量。

            .. note::

                由于 ``python`` 语言的特性，设置该参数后会导致相应倍数的内存增长，这可能会对您程序的执行带来一定的影响。另外，使用多进程时，
                ``func`` 函数中的打印将不会输出。

        :param func: 用户自定义处理函数，参数是 ``DataSet`` 中的 ``Instance``
        :param _apply_field: 需要传进去func的数据集的field_name
        :param progress_bar: 显示 progress_bar 的方式，支持 `["rich", "tqdm", None]`。
        :param progress_desc: 进度条的描述字符，默认为'Main
        """
        if isinstance(func, LambdaType) and num_proc>1 and func.__name__ == "<lambda>":
            raise TypeError("Lambda function does not support multiple processes, please set `num_proc=0`.")
        if num_proc>1 and sys.platform in ('win32', 'msys', 'cygwin'):
            raise RuntimeError("Your platform does not support multiprocessing with fork, please set `num_proc=0`")

        if num_proc < 2:
            results = _apply_single(ds=self, _apply_field=_apply_field, func=func,
                                    desc=progress_desc, progress_bar=progress_bar)
        else:
            # TODO 1. desc这个需要修改一下，应该把 subprocess 的 desc 修改一下。修改成Process 1 / Process 2
            import multiprocessing as mp
            ctx = mp.get_context('fork')
            num_proc = min(num_proc, len(self))
            #   划分数据集
            shard_len = len(self) // num_proc
            num_left_sample = len(self) % num_proc
            start = 0
            shard_data = []
            for _i in range(num_proc):
                end = shard_len + int(_i<num_left_sample) + start
                shard_data.append(self[start:end])
                start = end
            #   配置共享参数，线程以实现 main progress 能够实时更新。
            counter = ctx.Value('i', 0, lock=True)
            pool = []
            queues = []
            results = []
            for i in range(num_proc):
                queue = ctx.SimpleQueue()
                proc = ctx.Process(target=_multi_proc, args=(shard_data[i], _apply_field, func, counter, queue))
                proc.start()
                pool.append(proc)
                queues.append(queue)
            progress_bar = progress_bars.get(progress_bar, DummyFRichProgress())
            total_len = len(self)
            task_id = progress_bar.add_task(description=progress_desc, total=total_len)
            last_count = -1
            while counter.value < total_len or last_count == -1:
                while counter.value == last_count:
                    time.sleep(0.1)
                advance = counter.value - last_count
                last_count = counter.value
                progress_bar.update(task_id, advance=advance, refresh=True)

            for idx, proc in enumerate(pool):
                results.extend(pickle.loads(queues[idx].get()))
                proc.join()
            progress_bar.destroy_task(task_id)
        return results

    def apply_more(self, func: Callable = None, modify_fields: bool = True,
                   num_proc: int = 0, progress_desc: str = '', progress_bar: str = 'rich'):
        r"""
        将 ``DataSet`` 中每个 ``Instance`` 传入到 ``func`` 中，并获取它的返回值。``func`` 可以返回一个或多个 field 上的结果。

        .. note::
            ``apply_more`` 与 ``apply`` 的区别：

            1. ``apply_more`` 可以返回多个 field 的结果， ``apply`` 只可以返回一个field 的结果；

            2. ``apply_more`` 的返回值是一个字典，每个 key-value 对中的 key 表示 field 的名字，value 表示计算结果；

            3. ``apply_more`` 默认修改 ``DataSet`` 中的 field ，``apply`` 默认不修改。

        :param func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值是一个字典，key 是field 的名字，value 是对应的结果
        :param modify_fields: 是否用结果修改 ``DataSet`` 中的 ``Field`` ， 默认为 ``True``
        :param num_proc: 使用进程的数量。

            .. note::

                由于 ``python`` 语言的特性，设置该参数后会导致相应倍数的内存增长，这可能会对您程序的执行带来一定的影响。另外，使用多进程时，
                ``func`` 函数中的打印将不会输出。

        :param progress_desc: 当 progress_bar 不为 ``None`` 时，可以显示当前正在处理的进度条名称
        :param progress_bar: 显示进度条的方式，支持 ``["rich", "tqdm", None]``。
        :return: 一个字典
        """
        assert callable(func), "The func is not callable."
        assert len(self) != 0, "Null DataSet cannot use apply()."
        assert num_proc >= 0, "num_proc must >= 0"
        idx = -1

        results = {}
        apply_out = self._apply_process(num_proc, func, progress_desc=progress_desc,
                                        progress_bar=progress_bar)
        #   只检测第一个数据是否为dict类型，若是则默认所有返回值为dict；否则报错。
        if not isinstance(apply_out[0], Mapping):
            raise Exception(f"The result of func:{_get_fun_msg(func)} is not a dict, but of type {type(apply_out[0])}")

        for key, value in apply_out[0].items():
            results[key] = [value]
        #   尝试合并所有dict数据, idx+1 的原因是第一条数据不可能出现错误，已经将第一条数据取出来
        try:
            for idx, per_out in enumerate(apply_out[1:]):
                if len(set(results.keys()) - set(per_out.keys())):
                    raise ApplyResultException(f"Apply results have different fields:{set(results.keys())} and "
                                               f"{set(per_out.keys())}", idx + 1)
                for key, value in per_out.items():
                    results[key].append(value)

        except Exception as e:
            if idx != -1:
                logger.error("Exception happens at the `{}`th instance.".format(idx + 1))
            raise e

        if modify_fields is True:
            for field, result in results.items():
                self.add_field(field_name=field, fields=result)

        return results

    def apply(self, func: Callable = None, new_field_name: str = None,
              num_proc: int = 0, progress_bar: str = 'rich', progress_desc: str = ''):
        """
        将 ``DataSet`` 中每个 ``Instance`` 传入到 ``func`` 中，并获取它的返回值。``func`` 仅能返回一个结果。

        :param func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值将被写入 ``new_field_name`` 中。
        :param new_field_name: 将 ``func`` 返回的内容放入到 ``new_field_name`` 这个 field中 ，如果名称与已有的 field 相同，则覆
            盖之前的 field。如果为 ``None`` 则不创建新的 field。
        :param num_proc: 使用进程的数量。

            .. note::

                由于 ``python`` 语言的特性，设置该参数后会导致相应倍数的内存增长，这可能会对您程序的执行带来一定的影响。另外，使用多进程时，
                ``func`` 函数中的打印将不会输出。

        :param progress_bar: 显示进度条的方式，支持 ``["rich", "tqdm", None]``。
        :param progress_desc: 如果不为 ``None``，则会显示当前正在处理的进度条的名称。
        """
        assert callable(func), "The func you provide is not callable."
        assert len(self) != 0, "Null DataSet cannot use apply()."
        assert num_proc >= 0, "num_proc must be an integer >= 0."
        try:
            results = self._apply_process(num_proc=num_proc, func=func, progress_bar=progress_bar,
                                          progress_desc=progress_desc)
        except BaseException as e:
            raise e

        if new_field_name is not None:
            self.add_field(field_name=new_field_name, fields=results)

        return results

    def add_seq_len(self, field_name: str, new_field_name='seq_len'):
        r"""
        将使用 :func:`len` 直接对 ``field_name`` 中每个元素作用，将其结果作为 sequence length, 并放入 ``new_field_name`` 这个 field。

        :param field_name: 需要处理的 field_name
        :param new_field_name: 新的 field_name
        :return:
        """
        if self.has_field(field_name=field_name):
            self.apply_field(len, field_name, new_field_name=new_field_name)
        else:
            raise KeyError(f"Field:{field_name} not found.")
        return self

    def drop(self, func: Callable, inplace=True):
        r"""
        删除某些 Instance。 需要注意的是 ``func`` 接受一个 Instance ，返回 bool 值。返回值为 ``True`` 时，
        该 Instance 会被移除或者不会包含在返回的 DataSet 中。

        :param func: 接受一个 Instance 作为参数，返回 bool 值。为 ``True`` 时删除该 instance
        :param inplace: 是否在当前 DataSet 中直接删除 instance；如果为 False，将返回一个新的 DataSet。

        :return: DataSet
        """
        if inplace:
            results = [ins for ins in self if not func(ins)]
            for name, old_field in self.field_arrays.items():
                self.field_arrays[name].content = [ins[name] for ins in results]
            return self
        else:
            results = [ins for ins in self if not func(ins)]
            if len(results) != 0:
                dataset = DataSet(results)
                return dataset
            else:
                return DataSet()

    def split(self, ratio: float, shuffle=True):
        r"""
        将 DataSet 按照 ``ratio`` 的比例拆分，返回两个 DataSet

        :param ratio: 0<ratio<1, 返回的第一个 DataSet 拥有 ``ratio`` 比例的数据，第二个 DataSet 拥有 ``1-ratio`` 的数据；
        :param shuffle: 在拆分前是否进行排序。为 False，返回的第一个 dataset 就是当前 dataset 中前 ``ratio`` 比例的数据；
        :return: 拆分后的两个 DataSet；
        """
        assert len(self) > 1, f'DataSet with {len(self)} instance cannot be split.'
        assert isinstance(ratio, float)
        assert 0 < ratio < 1
        all_indices = [_ for _ in range(len(self))]
        if shuffle:
            np.random.shuffle(all_indices)
        split = int(ratio * len(self))
        if split == 0:
            error_msg = f'Dev DataSet has `{split}` instance after split.'
            raise IndexError(error_msg)
        dev_indices = all_indices[:split]
        train_indices = all_indices[split:]
        dev_set = DataSet()
        train_set = DataSet()
        for idx in dev_indices:
            dev_set.append(self[idx])
        for idx in train_indices:
            train_set.append(self[idx])
        dev_set._collator = deepcopy(self.collator)
        train_set._collator = deepcopy(self.collator)

        return dev_set, train_set

    def save(self, path: str) -> None:
        r"""
        保存 DataSet。

        :param path: 保存路径；
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        r"""
        从保存的 DataSet pickle 文件的路径中读取 DataSet

        :param path: 读取路径；
        :return: 读取出的 DataSet
        """
        with open(path, 'rb') as f:
            d = pickle.load(f)
            assert isinstance(d, DataSet), "The object is not DataSet, but {}.".format(type(d))
        return d

    def concat(self, dataset: 'DataSet', inplace:bool=True, field_mapping:Dict=None) -> 'DataSet':
        """
        将当前 DataSet 与输入的 ``dataset`` 结合成一个更大的 dataset，需要保证两个 dataset 都包含了相同的 field。结合后的 dataset
        的 field_name 和 _collator 以当前 dataset 为准。若 ``dataset`` 中包含的 field 多于当前的 DataSet，则多余的 field 会被忽略；
        若 ``dataset`` 中未包含所有当前 DataSet 含有 field，则会报错。

        :param dataset: 需要和当前 DataSet 拼接的 ``dataset``；
        :param inplace: 是否直接将 ``dataset`` 组合到当前 DataSet 中；
        :param field_mapping: 当传入的 ``dataset`` 中的 field 名称和当前 dataset 不一致时，需要通过 ``field_mapping`` 把输入的 ``dataset``
            中的 field 名称映射到当前 field。``field_mapping`` 为 dict 类型，key 为 11dataset`` 中的 field 名称，value 是需要映射成的名称

        :return: :class:`~fastNLP.core.dataset.DataSet`
        """
        assert isinstance(dataset, DataSet), "Can only concat two datasets."

        fns_in_this_dataset = set(self.get_field_names())
        fns_in_other_dataset = dataset.get_field_names()
        reverse_field_mapping = {}
        if field_mapping is not None:
            fns_in_other_dataset = [field_mapping.get(fn, fn) for fn in fns_in_other_dataset]
            reverse_field_mapping = {v: k for k, v in field_mapping.items()}
        fns_in_other_dataset = set(fns_in_other_dataset)
        fn_not_seen = list(fns_in_this_dataset - fns_in_other_dataset)

        if fn_not_seen:
            raise RuntimeError(f"The following fields are not provided in the dataset:{fn_not_seen}")

        if inplace:
            ds = self
        else:
            ds = deepcopy(self)

        for fn in fns_in_this_dataset:
            ds.get_field(fn).content.extend(deepcopy(dataset.get_field(reverse_field_mapping.get(fn, fn)).content))

        return ds

    @classmethod
    def from_pandas(cls, df):
        """
        从 :class:`pandas.DataFrame` 中读取并数据转化为 DataSet

        :param df: 使用 pandas 读取的数据
        :return:
        """
        df_dict = df.to_dict(orient='list')
        return cls(df_dict)

    def to_pandas(self):
        """
        将 DataSet 数据转为 :class:`pandas.DataFrame` 类型的数据

        :return:
        """
        import pandas as pd
        dict_ = {key: value.content for key, value in self.field_arrays.items()}
        return pd.DataFrame.from_dict(dict_)

    def to_csv(self, path: str):
        """
        将 DataSet 保存为 csv 文件

        :param path: 保存到路径
        :return:
        """

        df = self.to_pandas()
        return df.to_csv(path, encoding="utf-8")

    @property
    def collator(self) -> Collator:
        if self._collator is None:
            self._collator = Collator()
        return self._collator

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
            batch 形式。 Collator 将自动 unbatch 数据，然后将各个 field 组成各自的 batch 。
        :return: 自身的 collator；
        """
        if isinstance(self.collator, Collator):
            self.collator.set_pad(field_name=field_name, pad_val=pad_val, dtype=dtype, pad_fn=pad_fn, backend=backend)
            return self.collator
        else:
            raise ValueError(f"Only when the collate_fn is a fastNLP Collator, set_pad() is allowed.")

    def set_ignore(self, *field_names) -> Collator:
        """
        ``DataSet`` 中想要对绑定的 collator 进行调整可以调用此函数。 ``collator`` 为 :class:`~fastNLP.core.collators.Collator`
        时该函数才有效。调用该函数可以设置忽略输出某些 field 的内容，被设置的 field 将在 batch 的输出中被忽略::

            dataset.set_ignore('field1', 'field2')

        :param field_names: field_name: 需要调整的 field 的名称。如果 :meth:`Dataset.__getitem__` 方法返回的是字典类型，则可以直接使用对应的
            field 的 key 来表示，如果是嵌套字典，可以使用元组表示多层次的 key，例如 ``{'a': {'b': 1}}`` 中可以使用 ``('a', 'b')``;
            如果 :meth:`Dataset.__getitem__` 返回的是 Sequence 类型，则可以使用 ``'_0'``, ``'_1'`` 表示序列中第 **0** 或 **1** 个元素。
        :return: 自身的 collator；
        """
        if isinstance(self.collator, Collator):
            self.collator.set_ignore(*field_names)
            return self.collator
        else:
            raise ValueError(f"Only when the collate_fn is a fastNLP Collator, set_ignore() is allowed.")

    @classmethod
    def from_datasets(cls, dataset):
        """
        将 Huggingface Dataset 转为 fastNLP 的 DataSet

        :param dataset 为实例化好的 huggingface Dataset 对象
        """
        from datasets import Dataset
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Support huggingface dataset, but is {type(dataset)}!")

        data_dict = dataset.to_dict()
        return DataSet(data_dict)