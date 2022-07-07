"""
:class:`DataBundle` 是 **fastNLP** 提供的用于方便快捷地管理多个数据集的工具，并有诸多接口来进行批量的数据处理。
"""

__all__ = [
    'DataBundle',
]

from typing import Union, List, Callable

from ..core.dataset import DataSet
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core import logger


class DataBundle:
    r"""
    经过处理的数据信息，包括一系列数据集（比如：分开的训练集、验证集和测试集）以及各个 field 对应的 vocabulary。该对象一般由
    **fastNLP** 中各种 :class:`~fastNLP.io.loader.Loader` 的 :meth:`load` 函数生成，可以通过以下的方法获取里面的内容::

        data_bundle = YelpLoader().load({'train':'/path/to/train', 'dev': '/path/to/dev'})
        train_vocabs = data_bundle.vocabs['train']
        train_data = data_bundle.datasets['train']
        dev_data = data_bundle.datasets['train']

    :param vocabs: 从名称（字符串）到 :class:`~fastNLP.core.Vocabulary` 类型的字典
    :param datasets: 从名称（字符串）到 :class:`~fastNLP.core.dataset.DataSet` 类型的字典。建议不要将相同的 ``DataSet`` 对象重复传入，
        否则可能会在使用 :class:`~fastNLP.io.pipe.Pipe` 处理数据的时候遇到问题，若多个数据集确需一致，请手动 ``deepcopy`` 后传入。
    """

    def __init__(self, vocabs=None, datasets=None):
        self.vocabs = vocabs or {}
        self.datasets = datasets or {}

    def set_vocab(self, vocab: Vocabulary, field_name: str):
        r"""
        向 :class:`DataBunlde` 中增加 ``vocab``

        :param vocab: :class:`~fastNLP.core.Vocabulary` 类型的词表
        :param field_name: 这个 vocab 对应的 field 名称
        :return: self
        """
        assert isinstance(vocab, Vocabulary), "Only fastNLP.core.Vocabulary supports."
        self.vocabs[field_name] = vocab
        return self

    def set_dataset(self, dataset: DataSet, name: str):
        r"""

        :param dataset: 传递给 :class:`DataBundle` 的 :class:`~fastNLP.core.dataset.DataSet` 
        :param name: ``dataset`` 的名称
        :return: self
        """
        assert isinstance(dataset, DataSet), "Only fastNLP.DataSet supports."
        self.datasets[name] = dataset
        return self

    def get_dataset(self, name: str) -> DataSet:
        r"""
        获取名为 ``name`` 的 dataset

        :param name: dataset的名称，一般为 'train', 'dev', 'test' 。
        :return:
        """
        if name in self.datasets.keys():
            return self.datasets[name]
        else:
            error_msg = f'DataBundle do NOT have DataSet named {name}. ' \
                        f'It should be one of {self.datasets.keys()}.'
            logger.error(error_msg)
            raise KeyError(error_msg)

    def delete_dataset(self, name: str):
        r"""
        删除名为 ``name`` 的 dataset

        :param name:
        :return: self
        """
        self.datasets.pop(name, None)
        return self

    def get_vocab(self, name: str) -> Vocabulary:
        r"""
        获取 field 名为 ``field_name`` 对应的词表

        :param field_name: 名称
        :return: :class:`~fastNLP.core.Vocabulary`
        """
        if name in self.vocabs.keys():
            return self.vocabs[name]
        else:
            error_msg = f'DataBundle do NOT have Vocabulary named {name}. ' \
                        f'It should be one of {self.vocabs.keys()}.'
            logger.error(error_msg)
            raise KeyError(error_msg)

    def delete_vocab(self, field_name: str):
        r"""
        删除名为 ``field_name`` 的 vocab

        :param field_name:
        :return: self
        """
        self.vocabs.pop(field_name, None)
        return self

    @property
    def num_dataset(self):
        return len(self.datasets)

    @property
    def num_vocab(self):
        return len(self.vocabs)

    def copy_field(self, field_name: str, new_field_name: str, ignore_miss_dataset: bool=True):
        r"""
        将所有的 dataset 中名为 ``field_name`` 的 Field 复制一份并命名为 ``new_field_name``。

        :param field_name:
        :param new_field_name:
        :param ignore_miss_dataset: 如果为 ``True`` ，则当 ``field_name`` 在某个 dataset 内不存在时，直接忽略该 dataset，
            如果为 ``False`` 则会报错。
        :return: self
        """
        for name, dataset in self.datasets.items():
            if dataset.has_field(field_name=field_name):
                dataset.copy_field(field_name=field_name, new_field_name=new_field_name)
            elif not ignore_miss_dataset:
                raise KeyError(f"{field_name} not found DataSet:{name}.")
        return self

    def rename_field(self, field_name: str, new_field_name: str, ignore_miss_dataset: bool=True, rename_vocab: bool=True):
        r"""
        将所有的 dataset 中名为 ``field_name`` 的 Field 重命名为 ``new_field_name``。

        :param field_name:
        :param new_field_name:
        :param ignore_miss_dataset: 如果为 ``True`` ，则当 ``field_name`` 在某个 dataset 内不存在时，直接忽略该 dataset，
            如果为 ``False`` 则会报错。
        :param rename_vocab: 如果该 ``field_name`` 同时也存在于 vocabs 中，则也会进行重命名
        :return: self
        """
        for name, dataset in self.datasets.items():
            if dataset.has_field(field_name=field_name):
                dataset.rename_field(field_name=field_name, new_field_name=new_field_name)
            elif not ignore_miss_dataset:
                raise KeyError(f"{field_name} not found DataSet:{name}.")
        if rename_vocab:
            if field_name in self.vocabs:
                self.vocabs[new_field_name] = self.vocabs.pop(field_name)

        return self

    def delete_field(self, field_name: str, ignore_miss_dataset: bool=True, delete_vocab: bool=True):
        r"""
        将所有的 dataset 中名为 ``field_name`` 的 Field 删除。

        :param field_name:
        :param ignore_miss_dataset: 如果为 ``True`` ，则当 ``field_name`` 在某个 dataset 内不存在时，直接忽略该 dataset，
            如果为 ``False`` 则会报错。
        :param delete_vocab: 如果该 ``field_name`` 也在 vocabs 中存在，则也会删除。
        :return: self
        """
        for name, dataset in self.datasets.items():
            if dataset.has_field(field_name=field_name):
                dataset.delete_field(field_name=field_name)
            elif not ignore_miss_dataset:
                raise KeyError(f"{field_name} not found DataSet:{name}.")
        if delete_vocab:
            if field_name in self.vocabs:
                self.vocabs.pop(field_name)
        return self

    def iter_datasets(self) -> Union[str, DataSet]:
        r"""
        迭代 dataset

        Example::

            for name, dataset in data_bundle.iter_datasets():
                pass

        """
        for name, dataset in self.datasets.items():
            yield name, dataset

    def get_dataset_names(self) -> List[str]:
        r"""
        :return: 所有 dataset 的名称
        """
        return list(self.datasets.keys())

    def get_vocab_names(self) -> List[str]:
        r"""
        :return: 所有词表的名称
        """
        return list(self.vocabs.keys())

    def iter_vocabs(self):
        r"""
        迭代词表

        Example::

            for field_name, vocab in data_bundle.iter_vocabs():
                pass

        """
        for field_name, vocab in self.vocabs.items():
            yield field_name, vocab

    def apply_field(self, func: Callable, field_name: str, new_field_name: str, num_proc: int = 0,
                    ignore_miss_dataset: bool = True, progress_desc: str = '', progress_bar: str = 'rich'):
        r"""
        对 :class:`DataBundle` 中所有的 dataset 使用 :meth:`~fastNLP.core.dataset.DataSet.apply_field` 方法

        :param func: 对指定 field 进行处理的函数，注意其输入应为 ``instance`` 中名为 ``field_name`` 的 field 的内容；
        :param field_name: 传入 ``func`` 的 field 名称；
        :param new_field_name: 函数执行结果写入的 ``field`` 名称。该函数会将 ``func`` 返回的内容放入到 ``new_field_name`` 对
            应的 ``field`` 中，注意如果名称与已有的 field 相同则会进行覆盖。如果为 ``None`` 则不会覆盖和创建 field ；
        :param num_proc: 使用进程的数量。
        
            .. note::
            
                由于 ``python`` 语言的特性，设置该参数后会导致相应倍数的内存增长，这可能会对您程序的执行带来一定的影响。另外，使用多进程时，
                ``func`` 函数中的打印将不会输出。

        :param ignore_miss_dataset: 如果为 ``True`` ，则当 ``field_name`` 在某个 dataset 内不存在时，直接忽略该 dataset，
            如果为 ``False`` 则会报错。
        :param progress_desc: 如果不为 ``None``，则会显示当前正在处理的进度条的名称；
        :param progress_bar: 显示进度条的方式，支持 ``["rich", "tqdm", None]``。
        :return: self
        """
        _progress_desc = progress_desc
        for name, dataset in self.datasets.items():
            if len(_progress_desc) == 0:
                _progress_desc = 'Processing'
            progress_desc = _progress_desc + f' for `{name}`'
            if dataset.has_field(field_name=field_name):
                dataset.apply_field(func=func, field_name=field_name, new_field_name=new_field_name, num_proc=num_proc,
                                    progress_desc=progress_desc, progress_bar=progress_bar)
            elif not ignore_miss_dataset:
                raise KeyError(f"{field_name} not found DataSet:{name}.")
        return self

    def apply_field_more(self, func: Callable, field_name: str,  modify_fields: str=True, num_proc: int = 0,
                         ignore_miss_dataset=True, progress_bar: str = 'rich', progress_desc: str = ''):
        r"""
        对 :class:`DataBundle` 中所有的 dataset 使用 :meth:`~fastNLP.core.DataSet.apply_field_more` 方法

        .. note::
            ``apply_field_more`` 与 ``apply_field`` 的区别参考 :meth:`fastNLP.DataSet.apply_more` 中关于 ``apply_more`` 与
            ``apply`` 区别的介绍。

        :param func: 对指定 field 进行处理的函数，注意其输入应为 ``instance`` 中名为 ``field_name`` 的 field 的内容；
        :param field_name: 传入 ``func`` 的 field 名称；
        :param modify_fields: 是否用结果修改 ``DataSet`` 中的 ``Field`` ， 默认为 ``True``
        :param num_proc: 使用进程的数量。
        
            .. note::
            
                由于 ``python`` 语言的特性，设置该参数后会导致相应倍数的内存增长，这可能会对您程序的执行带来一定的影响。另外，使用多进程时，
                ``func`` 函数中的打印将不会输出。

        :param ignore_miss_dataset: 如果为 ``True`` ，则当 ``field_name`` 在某个 dataset 内不存在时，直接忽略该 dataset，
            如果为 ``False`` 则会报错。
        :param progress_desc: 如果不为 ``None``，则会显示当前正在处理的进度条的名称；
        :param progress_bar: 显示进度条的方式，支持 ``["rich", "tqdm", None]``。
        :return: 一个字典套字典，第一层的 key 是 dataset 的名字，第二层的 key 是 field 的名字
        """
        res = {}
        _progress_desc = progress_desc
        for name, dataset in self.datasets.items():
            if len(_progress_desc) == 0:
                _progress_desc = 'Processing'
            progress_desc = _progress_desc + f' for `{name}`'
            if dataset.has_field(field_name=field_name):
                res[name] = dataset.apply_field_more(func=func, field_name=field_name, num_proc=num_proc,
                                                     modify_fields=modify_fields,
                                                     progress_bar=progress_bar, progress_desc=progress_desc)
            elif not ignore_miss_dataset:
                raise KeyError(f"{field_name} not found DataSet:{name} .")
        return res

    def apply(self, func: Callable, new_field_name: str, num_proc: int = 0,
              progress_desc: str = '', progress_bar: bool = True):
        r"""
        对 :class:`~DataBundle` 中所有的 dataset 使用 :meth:`~fastNLP.core.DataSet.apply` 方法

        :param func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值是一个字典，key 是field 的名字，value 是对应的结果
        :param new_field_name: 将 ``func`` 返回的内容放入到 ``new_field_name`` 这个 field中 ，如果名称与已有的 field 相同，则覆
            盖之前的 field。如果为 ``None`` 则不创建新的 field。
        :param num_proc: 使用进程的数量。

            .. note::

                由于 ``python`` 语言的特性，设置该参数后会导致相应倍数的内存增长，这可能会对您程序的执行带来一定的影响。另外，使用多进程时，
                ``func`` 函数中的打印将不会输出。

        :param progress_bar: 显示进度条的方式，支持 ``["rich", "tqdm", None]``。
        :param progress_desc: 如果不为 ``None``，则会显示当前正在处理的进度条的名称。
        :return: self
        """
        _progress_desc = progress_desc
        for name, dataset in self.datasets.items():
            if len(_progress_desc) == 0:
                _progress_desc = 'Processing'
            progress_desc = _progress_desc + f' for `{name}`'
            dataset.apply(func, new_field_name=new_field_name, num_proc=num_proc, progress_bar=progress_bar,
                          progress_desc=progress_desc)
        return self

    def apply_more(self, func: Callable, modify_fields: bool=True, num_proc: int = 0,
                   progress_desc: str = '', progress_bar: str = 'rich'):
        r"""
        对 :class:`~fastNLP.io.DataBundle` 中所有的 dataset 使用 :meth:`~fastNLP.DataSet.apply_more` 方法

        .. note::
            ``apply_more`` 与 ``apply`` 的区别参考 :meth:`fastNLP.DataSet.apply_more` 中关于 ``apply_more`` 与
            ``apply`` 区别的介绍。

        :param func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值是一个字典，key 是field 的名字，value 是对应的结果
        :param modify_fields: 是否用结果修改 ``DataSet`` 中的 ``Field`` ， 默认为 ``True``
        :param num_proc: 使用进程的数量。

            .. note::

                由于 ``python`` 语言的特性，设置该参数后会导致相应倍数的内存增长，这可能会对您程序的执行带来一定的影响。另外，使用多进程时，
                ``func`` 函数中的打印将不会输出。

        :param progress_desc: 当 progress_bar 不为 ``None`` 时，可以显示当前正在处理的进度条名称
        :param progress_bar: 显示进度条的方式，支持 ``["rich", "tqdm", None]``。

        :return: 一个字典套字典，第一层的 key 是 dataset 的名字，第二层的 key 是 field 的名字
        """
        res = {}
        _progress_desc = progress_desc
        for name, dataset in self.datasets.items():
            if len(_progress_desc) == 0:
                _progress_desc = 'Processing'
            progress_desc = _progress_desc + f' for `{name}`'
            res[name] = dataset.apply_more(func, modify_fields=modify_fields, num_proc=num_proc,
                                           progress_bar=progress_bar, progress_desc=progress_desc)
        return res

    def set_pad(self, field_name, pad_val=0, dtype=None, backend=None, pad_fn=None) -> "DataBundle":
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
        :return: self
        """
        for _, ds in self.iter_datasets():
            ds.collator.set_pad(field_name=field_name, pad_val=pad_val, dtype=dtype, backend=backend,
                                pad_fn=pad_fn)
        return self

    def set_ignore(self, *field_names) -> "DataBundle":
        """
        ``DataSet`` 中想要对绑定的 collator 进行调整可以调用此函数。 ``collator`` 为 :class:`~fastNLP.core.collators.Collator`
        时该函数才有效。调用该函数可以设置忽略输出某些 field 的内容，被设置的 field 将在 batch 的输出中被忽略::

            databundle.set_ignore('field1', 'field2')

        :param field_names: field_name: 需要调整的 field 的名称。如果 :meth:`Dataset.__getitem__` 方法返回的是字典类型，则可以直接使用对应的
            field 的 key 来表示，如果是嵌套字典，可以使用元组表示多层次的 key，例如 ``{'a': {'b': 1}}`` 中可以使用 ``('a', 'b')``;
            如果 :meth:`Dataset.__getitem__` 返回的是 Sequence 类型，则可以使用 ``'_0'``, ``'_1'`` 表示序列中第 **0** 或 **1** 个元素。
        :return: self
        """
        for _, ds in self.iter_datasets():
            ds.collator.set_ignore(*field_names)
        return self

    def __repr__(self) -> str:
        _str = ''
        if len(self.datasets):
            _str += 'In total {} datasets:\n'.format(self.num_dataset)
            for name, dataset in self.datasets.items():
                _str += '\t{} has {} instances.\n'.format(name, len(dataset))
        if len(self.vocabs):
            _str += 'In total {} vocabs:\n'.format(self.num_vocab)
            for name, vocab in self.vocabs.items():
                _str += '\t{} has {} entries.\n'.format(name, len(vocab))
        return _str
