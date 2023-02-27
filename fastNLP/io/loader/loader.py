from typing import Dict, Union

from fastNLP.core.dataset import DataSet
from fastNLP.io.data_bundle import DataBundle
from fastNLP.io.file_utils import _get_dataset_url, cached_path, get_cache_path
from fastNLP.io.utils import check_loader_paths

__all__ = ['Loader']


class Loader:
    r"""
    **fastNLP** 读取数据的模块。

    ``Loader`` 用于读取数据，并将内容读取到 :class:`.DataSet` 或者 :class:`.\
    DataBundle` 中。所有的 ``Loader`` 都支持以下的三个方法：:meth:`__init__`，
    :meth:`_load` , :meth:`loads`。其中 ``__init__(...)`` 用于申明读取参数，
    以及说明该 ``Loader`` 支持的数据格式、读取后 :class:`.DataSet` 中的
    `field` ； ``_load(path)`` 方法传入文件路径读取单个文件，并返回 :class:`.\
    DataSet`，返回的 DataSet 的内容可以通过每个 ``Loader`` 的文档判断出；``load
    (paths)`` 读取文件夹下的文件，并返回 :class:`.DataBundle` 类型的对象。
    :meth:`_load` 方法支持以下几种类型的参数：

    - 传入 ``None``：
      将尝试自动下载数据集并缓存。但不是所有的数据都可以直接下载。
    - 传入一个文件的 path：
      返回的 `data_bundle` 包含一个名为 `train` 的 dataset，可以通 ``data_bundle.
      get_dataset('train')`` 获取。
    - 传入一个文件夹目录：
      将读取的是这个文件夹下文件名中包含 `train`、`test`、`dev` 的文件，其它文件
      会被忽略。假设某个目录下的文件为::

            |
            +-train.txt
            +-dev.txt
            +-test.txt
            +-other.txt

      在 ``Loader().load('/path/to/dir')`` 返回的 `data_bundle` 中可以用
      ``data_bundle.get_dataset('train')``、``data_bundle.get_dataset
      ('dev')``、``data_bundle.get_dataset('test')`` 获取对应的 `dataset` ，
      其中 `other.txt` 的内容会被忽略。假设某个目录下的文件为::

            |
            +-train.txt
            +-dev.txt

      在 ``Loader().load('/path/to/dir')`` 返回的 `data_bundle` 中可以用
      ``data_bundle.get_dataset('train')``、``data_bundle.get_dataset('dev')``
      获取对应的 dataset。

    - 传入一个字典：
      字典的的 key 为 ``dataset`` 的名称，value 是该 ``dataset`` 的文件路径::

            paths = {
                'train':'/path/to/train',
                'dev': '/path/to/dev',
                'test':'/path/to/test'
            }

      在 ``Loader().load(paths)``  返回的 `data_bundle` 中可以用 ``data_bundle.
      get_dataset('train')``、``data_bundle.get_dataset('dev')``、
      ``data_bundle.get_dataset('test')`` 来获取对应的 `dataset`。

    除此之外还有 :meth:`download` 函数：自动将该数据集下载到缓存地址，默认缓存地址为
    ``~/.fastNLP/datasets/``。由于版权等原因，不是所有的 ``Loader`` 都实现了该方
    法。该方法会返回下载后文件所处的缓存地址。
    """

    def __init__(self):
        pass

    def _load(self, path: str) -> DataSet:
        r"""
        给定一个路径，返回读取的 :class:`~fastNLP.core.DataSet`。

        :param path: 路径
        :return: :class:`~fastNLP.core.DataSet`
        """
        raise NotImplementedError

    def load(self,
             paths: Union[str, Dict[str, str], None] = None) -> DataBundle:
        r"""
        从指定一个或多个路径中的文件中读取数据，返回 :class:`~fastNLP.io.\
        DataBundle`。

        :param paths: 支持以下的几种输入方式：

            - ``None`` -- 先查看本地是否有缓存，如果没有则自动下载并缓存。
            - 一个目录，该目录下名称包含 ``'train'`` 的被认为是训练集，包含
              ``'test'`` 的被认为是测试集，包含 ``'dev'`` 的被认为是验证集 / 开发
              集，如果检测到多个文件名包含 ``'train'``、 ``'dev'``、 ``'test'`` 则
              会报错::


                data_bundle = xxxLoader().load('/path/to/dir')
                # 返回的 DataBundle 中 datasets 根据目录下是否检测到 train
                # dev、 test 等有所变化，可以通过以下的方式取出 DataSet
                tr_data = data_bundle.get_dataset('train')
                # 如果目录下有文件包含test这个字段
                te_data = data_bundle.get_dataset('test')

            - 传入一个 :class:`dict`，比如训练集、验证集和测试集不在同一个目录下，或
              者名称中不包含 ``'train'``、 ``'dev'``、 ``'test'`` ::

                paths = {
                    'train':"/path/to/tr_data.conll",
                    'dev':"/to/validate.conll",
                    "test":"/to/te_data.conll"
                }
                data_bundle = xxxLoader().load(paths)
                # 返回的 DataBundle 中的 dataset 中包含"train", "dev", "test"
                dev_data = data_bundle.get_dataset('dev')

            - 传入文件路径::

                # 返回DataBundle对象，datasets中仅包含'train'
                data_bundle = xxxLoader().load("/path/to/a/train.conll")
                tr_data = data_bundle.get_dataset('train')  # 取出DataSet

        :return: :class:`~fastNLP.io.DataBundle`
        """
        if paths is None:
            paths = self.download()
        paths = check_loader_paths(paths)
        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle

    def download(self) -> str:
        r"""
        自动下载该数据集

        :return: 下载后解压目录
        """
        raise NotImplementedError(
            f'{self.__class__} cannot download data automatically.')

    @staticmethod
    def _get_dataset_path(dataset_name):
        r"""
        传入dataset的名称，获取读取数据的目录。如果数据不存在，会尝试自动下载并缓存
        （如果支持的话）

        :param str dataset_name: 数据集的名称
        :return: str, 数据集的目录地址。直接到该目录下读取相应的数据即可。
        """

        default_cache_path = get_cache_path()
        url = _get_dataset_url(dataset_name)
        output_dir = cached_path(
            url_or_filename=url, cache_dir=default_cache_path, name='dataset')

        return output_dir
