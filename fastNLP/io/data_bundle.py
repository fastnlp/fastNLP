r"""
.. todo::
    doc
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
    经过处理的数据信息，包括一系列数据集（比如：分开的训练集、验证集和测试集）以及各个field对应的vocabulary。该对象一般由fastNLP中各种
    Loader的load函数生成，可以通过以下的方法获取里面的内容

    Example::

        data_bundle = YelpLoader().load({'train':'/path/to/train', 'dev': '/path/to/dev'})
        train_vocabs = data_bundle.vocabs['train']
        train_data = data_bundle.datasets['train']
        dev_data = data_bundle.datasets['train']

    """

    def __init__(self, vocabs=None, datasets=None):
        r"""

        :param vocabs: 从名称(字符串)到 :class:`~fastNLP.Vocabulary` 类型的dict
        :param datasets: 从名称(字符串)到 :class:`~fastNLP.DataSet` 类型的dict。建议不要将相同的DataSet对象重复传入，可能会在
            使用Pipe处理数据的时候遇到问题，若多个数据集确需一致，请手动deepcopy后传入。
        """
        self.vocabs = vocabs or {}
        self.datasets = datasets or {}

    def set_vocab(self, vocab: Vocabulary, field_name: str):
        r"""
        向DataBunlde中增加vocab

        :param ~fastNLP.Vocabulary vocab: 词表
        :param str field_name: 这个vocab对应的field名称
        :return: self
        """
        assert isinstance(vocab, Vocabulary), "Only fastNLP.Vocabulary supports."
        self.vocabs[field_name] = vocab
        return self

    def set_dataset(self, dataset: DataSet, name: str):
        r"""

        :param ~fastNLP.DataSet dataset: 传递给DataBundle的DataSet
        :param str name: dataset的名称
        :return: self
        """
        assert isinstance(dataset, DataSet), "Only fastNLP.DataSet supports."
        self.datasets[name] = dataset
        return self

    def get_dataset(self, name: str) -> DataSet:
        r"""
        获取名为name的dataset

        :param str name: dataset的名称，一般为'train', 'dev', 'test'
        :return: DataSet
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
        删除名为name的DataSet

        :param str name:
        :return: self
        """
        self.datasets.pop(name, None)
        return self

    def get_vocab(self, field_name: str) -> Vocabulary:
        r"""
        获取field名为field_name对应的vocab

        :param str field_name: 名称
        :return: Vocabulary
        """
        if field_name in self.vocabs.keys():
            return self.vocabs[field_name]
        else:
            error_msg = f'DataBundle do NOT have Vocabulary named {field_name}. ' \
                        f'It should be one of {self.vocabs.keys()}.'
            logger.error(error_msg)
            raise KeyError(error_msg)

    def delete_vocab(self, field_name: str):
        r"""
        删除vocab
        :param str field_name:
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

    def copy_field(self, field_name: str, new_field_name: str, ignore_miss_dataset=True):
        r"""
        将DataBundle中所有的DataSet中名为field_name的Field复制一份并命名为叫new_field_name.

        :param str field_name:
        :param str new_field_name:
        :param bool ignore_miss_dataset: 当某个field名称在某个dataset不存在时，如果为True，则直接忽略该DataSet;
            如果为False，则报错
        :return: self
        """
        for name, dataset in self.datasets.items():
            if dataset.has_field(field_name=field_name):
                dataset.copy_field(field_name=field_name, new_field_name=new_field_name)
            elif not ignore_miss_dataset:
                raise KeyError(f"{field_name} not found DataSet:{name}.")
        return self

    def rename_field(self, field_name: str, new_field_name: str, ignore_miss_dataset=True, rename_vocab=True):
        r"""
        将DataBundle中所有DataSet中名为field_name的field重命名为new_field_name.

        :param str field_name:
        :param str new_field_name:
        :param bool ignore_miss_dataset: 当某个field名称在某个dataset不存在时，如果为True，则直接忽略该DataSet;
            如果为False，则报错
        :param bool rename_vocab: 如果该field同时也存在于vocabs中，会将该field的名称对应修改
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

    def delete_field(self, field_name: str, ignore_miss_dataset=True, delete_vocab=True):
        r"""
        将DataBundle中所有DataSet中名为field_name的field删除掉.

        :param str field_name:
        :param bool ignore_miss_dataset: 当某个field名称在某个dataset不存在时，如果为True，则直接忽略该DataSet;
            如果为False，则报错
        :param bool delete_vocab: 如果该field也在vocabs中存在，将该值也一并删除
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
        迭代data_bundle中的DataSet

        Example::

            for name, dataset in data_bundle.iter_datasets():
                pass

        :return:
        """
        for name, dataset in self.datasets.items():
            yield name, dataset

    def get_dataset_names(self) -> List[str]:
        r"""
        返回DataBundle中DataSet的名称

        :return:
        """
        return list(self.datasets.keys())

    def get_vocab_names(self) -> List[str]:
        r"""
        返回DataBundle中Vocabulary的名称

        :return:
        """
        return list(self.vocabs.keys())

    def iter_vocabs(self):
        r"""
        迭代data_bundle中的DataSet

        Example:

            for field_name, vocab in data_bundle.iter_vocabs():
                pass

        :return:
        """
        for field_name, vocab in self.vocabs.items():
            yield field_name, vocab

    def apply_field(self, func: Callable, field_name: str, new_field_name: str, num_proc: int = 0,
                    ignore_miss_dataset: bool = True, progress_desc: str = '', show_progress_bar: bool = True):
        r"""
        对 :class:`~fastNLP.io.DataBundle` 中所有的dataset使用 :meth:`~fastNLP.DataSet.apply_field` 方法

        :param callable func: input是instance中名为 `field_name` 的field的内容。
        :param str field_name: 传入func的是哪个field。
        :param str new_field_name: 将func返回的内容放入到 `new_field_name` 这个field中，如果名称与已有的field相同，则覆
            盖之前的field。如果为None则不创建新的field。
        :param bool ignore_miss_dataset: 当某个field名称在某个dataset不存在时，如果为True，则直接忽略该DataSet;
            如果为False，则报错
        :param ignore_miss_dataset:
        :param num_proc:
        :param progress_desc: 当show_progress_barm为True时，可以显示当前tqdm正在处理的名称
        :param show_progress_bar: 是否显示tqdm进度条

        """
        _progress_desc = progress_desc
        for name, dataset in self.datasets.items():
            if _progress_desc:
                progress_desc = _progress_desc + f' for `{name}`'
            if dataset.has_field(field_name=field_name):
                dataset.apply_field(func=func, field_name=field_name, new_field_name=new_field_name, num_proc=num_proc,
                                    progress_desc=progress_desc, show_progress_bar=show_progress_bar)
            elif not ignore_miss_dataset:
                raise KeyError(f"{field_name} not found DataSet:{name}.")
        return self

    def apply_field_more(self, func: Callable, field_name: str, num_proc: int = 0, modify_fields=True,
                         ignore_miss_dataset=True, progress_desc: str = '', show_progress_bar: bool = True):
        r"""
        对 :class:`~fastNLP.io.DataBundle` 中所有的 dataset 使用 :meth:`~fastNLP.DataSet.apply_field_more` 方法

        .. note::
            ``apply_field_more`` 与 ``apply_field`` 的区别参考 :meth:`fastNLP.DataSet.apply_more` 中关于 ``apply_more`` 与
            ``apply`` 区别的介绍。

        :param callable func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值是一个字典，key 是field 的名字，value 是对应的结果
        :param str field_name: 传入func的是哪个field。
        :param bool modify_fields: 是否用结果修改 `DataSet` 中的 `Field`， 默认为 True
        :param bool ignore_miss_dataset: 当某个field名称在某个dataset不存在时，如果为True，则直接忽略该DataSet;
            如果为False，则报错
        :param show_progress_bar: 是否显示tqdm进度条
        :param progress_desc: 当show_progress_barm为True时，可以显示当前tqdm正在处理的名称
        :param num_proc:

        :return Dict[str:Dict[str:Field]]: 返回一个字典套字典，第一层的 key 是 dataset 的名字，第二层的 key 是 field 的名字

        """
        res = {}
        _progress_desc = progress_desc
        for name, dataset in self.datasets.items():
            if _progress_desc:
                progress_desc = _progress_desc + f' for `{name}`'
            if dataset.has_field(field_name=field_name):
                res[name] = dataset.apply_field_more(func=func, field_name=field_name, num_proc=num_proc,
                                                     modify_fields=modify_fields,
                                                     show_progress_bar=show_progress_bar, progress_desc=progress_desc)
            elif not ignore_miss_dataset:
                raise KeyError(f"{field_name} not found DataSet:{name} .")
        return res

    def apply(self, func: Callable, new_field_name: str, num_proc: int = 0,
              progress_desc: str = '', show_progress_bar: bool = True, _apply_field: str = None):
        r"""
        对 :class:`~fastNLP.io.DataBundle` 中所有的 dataset 使用 :meth:`~fastNLP.DataSet.apply` 方法

        对DataBundle中所有的dataset使用apply方法

        :param callable func: input是instance中名为 `field_name` 的field的内容。
        :param str new_field_name: 将func返回的内容放入到 `new_field_name` 这个field中，如果名称与已有的field相同，则覆
            盖之前的field。如果为None则不创建新的field。
        :param _apply_field:
        :param show_progress_bar: 是否显示tqd进度条
        :param progress_desc: 当show_progress_bar为True时，可以显示当前tqd正在处理的名称
        :param num_proc:

        """
        _progress_desc = progress_desc
        for name, dataset in self.datasets.items():
            if _progress_desc:
                progress_desc = _progress_desc + f' for `{name}`'
            dataset.apply(func, new_field_name=new_field_name, num_proc=num_proc, show_progress_bar=show_progress_bar,
                          progress_desc=progress_desc, _apply_field=_apply_field)
        return self

    def apply_more(self, func: Callable, modify_fields=True, num_proc: int = 0,
                   progress_desc: str = '', show_progress_bar: bool = True):
        r"""
        对 :class:`~fastNLP.io.DataBundle` 中所有的 dataset 使用 :meth:`~fastNLP.DataSet.apply_more` 方法

        .. note::
            ``apply_more`` 与 ``apply`` 的区别参考 :meth:`fastNLP.DataSet.apply_more` 中关于 ``apply_more`` 与
            ``apply`` 区别的介绍。

        :param callable func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值是一个字典，key 是field 的名字，value 是对应的结果
        :param bool modify_fields: 是否用结果修改 ``DataSet`` 中的 ``Field`` ， 默认为 True
        :param show_progress_bar: 是否显示tqd进度条
        :param progress_desc: 当show_progress_bar为True时，可以显示当前tqd正在处理的名称
        :param num_proc:

        :return Dict[str:Dict[str:Field]]: 返回一个字典套字典，第一层的 key 是 dataset 的名字，第二层的 key 是 field 的名字
        """
        res = {}
        _progress_desc = progress_desc
        for name, dataset in self.datasets.items():
            if _progress_desc:
                progress_desc = _progress_desc + f' for `{name}`'
            res[name] = dataset.apply_more(func, modify_fields=modify_fields, num_proc=num_proc,
                                           show_progress_bar=show_progress_bar, progress_desc=progress_desc)
        return res

    def set_pad(self, field_name, pad_val=0, dtype=None, backend=None, pad_fn=None) -> "DataBundle":
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
        :return: self
        """
        for _, ds in self.iter_datasets():
            ds.collator.set_pad(field_name=field_name, pad_val=pad_val, dtype=dtype, backend=backend,
                                pad_fn=pad_fn)
        return self

    def set_ignore(self, *field_names) -> "DataBundle":
        """
        如果有的内容不希望输出，可以在此处进行设置，被设置的 field 将在 batch 的输出中被忽略。
        Example::

            collator.set_ignore('field1', 'field2')

        :param field_names: 需要忽略的 field 的名称。如果 Dataset 的 __getitem__ 方法返回的是 dict 类型的，则可以直接使用对应的
            field 的 key 来表示，如果是 nested 的 dict，可以使用元组来表示，例如 {'a': {'b': 1}} 中的使用 ('a', 'b'); 如果
            __getitem__ 返回的是 Sequence 类型的，则可以使用 '_0', '_1' 表示序列中第 0 或 1 个元素。
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

