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
# from ..core._logger import _logger


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
            print(error_msg)
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
            print(error_msg)
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
        对 :class:`~fastNLP.io.DataBundle` 中所有的dataset使用 :method:`~fastNLP.DataSet.apply_field` 方法

        :param callable func: input是instance中名为 `field_name` 的field的内容。
        :param str field_name: 传入func的是哪个field。
        :param str new_field_name: 将func返回的内容放入到 `new_field_name` 这个field中，如果名称与已有的field相同，则覆
            盖之前的field。如果为None则不创建新的field。
        :param bool ignore_miss_dataset: 当某个field名称在某个dataset不存在时，如果为True，则直接忽略该DataSet;
            如果为False，则报错
        :param ignore_miss_dataset:
        :param num_proc:
        :param progress_desc 当show_progress_barm为True时，可以显示当前tqdm正在处理的名称
        :param show_progress_bar 是否显示tqdm进度条

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
        对 :class:`~fastNLP.io.DataBundle` 中所有的 dataset 使用 :method:`~fastNLP.DataSet.apply_field_more` 方法

        .. note::
            ``apply_field_more`` 与 ``apply_field`` 的区别参考 :method:`fastNLP.DataSet.apply_more` 中关于 ``apply_more`` 与
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
        对 :class:`~fastNLP.io.DataBundle` 中所有的 dataset 使用 :method:`~fastNLP.DataSet.apply` 方法

        对DataBundle中所有的dataset使用apply方法

        :param callable func: input是instance中名为 `field_name` 的field的内容。
        :param str new_field_name: 将func返回的内容放入到 `new_field_name` 这个field中，如果名称与已有的field相同，则覆
            盖之前的field。如果为None则不创建新的field。
        :param _apply_field:
        :param show_progress_bar: 是否显示tqd进度条
        :param progress_desc: 当show_progress_bar为True时，可以显示当前tqd正在处理的名称
        :param num_proc

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
        对 :class:`~fastNLP.io.DataBundle` 中所有的 dataset 使用 :method:`~fastNLP.DataSet.apply_more` 方法

        .. note::
            ``apply_more`` 与 ``apply`` 的区别参考 :method:`fastNLP.DataSet.apply_more` 中关于 ``apply_more`` 与
            ``apply`` 区别的介绍。

        :param callable func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值是一个字典，key 是field 的名字，value 是对应的结果
        :param bool modify_fields: 是否用结果修改 ``DataSet`` 中的 ``Field`` ， 默认为 True
        :param show_progress_bar: 是否显示tqd进度条
        :param progress_desc: 当show_progress_bar为True时，可以显示当前tqd正在处理的名称
        :param num_proc

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

    def set_pad_val(self, *field_names, val=0) -> None:
        for _, ds in self.iter_datasets():
            ds.set_pad_val(*field_names, val=val)

    def set_input(self, *field_names) -> None:
        for _, ds in self.iter_datasets():
            ds.set_input(*field_names)

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

