"""undocumented"""

__all__ = [
    "Loader"
]

from typing import Union, Dict

from .. import DataBundle
from ..file_utils import _get_dataset_url, get_cache_path, cached_path
from ..utils import check_loader_paths
from ...core.dataset import DataSet


class Loader:
    """
    各种数据 Loader 的基类，提供了 API 的参考.
    
    """
    
    def __init__(self):
        pass
    
    def _load(self, path: str) -> DataSet:
        """
        给定一个路径，返回读取的DataSet。

        :param str path: 路径
        :return: DataSet
        """
        raise NotImplementedError
    
    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        """
        从指定一个或多个路径中的文件中读取数据，返回 :class:`~fastNLP.io.DataBundle` 。

        读取的field根据ConllLoader初始化时传入的headers决定。

        :param Union[str, Dict[str, str]] paths: 支持以下的几种输入方式
            (0) 如果为None，则先查看本地是否有缓存，如果没有则自动下载并缓存。

            (1) 传入一个目录, 该目录下名称包含train的被认为是train，包含test的被认为是test，包含dev的被认为是dev，如果检测到多个文件
                名包含'train'、 'dev'、 'test'则会报错::

                    data_bundle = ConllLoader().load('/path/to/dir')  # 返回的DataBundle中datasets根据目录下是否检测到train、
                    #  dev、 test等有所变化，可以通过以下的方式取出DataSet
                    tr_data = data_bundle.datasets['train']
                    te_data = data_bundle.datasets['test']  # 如果目录下有文件包含test这个字段

            (2) 传入文件路径::

                    data_bundle = ConllLoader().load("/path/to/a/train.conll") # 返回DataBundle对象, datasets中仅包含'train'
                    tr_data = data_bundle.datasets['train']  # 可以通过以下的方式取出DataSet

            (3) 传入一个dict，比如train，dev，test不在同一个目录下，或者名称中不包含train, dev, test::

                    paths = {'train':"/path/to/tr.conll", 'dev':"/to/validate.conll", "test":"/to/te.conll"}
                    data_bundle = ConllLoader().load(paths)  # 返回的DataBundle中的dataset中包含"train", "dev", "test"
                    dev_data = data_bundle.datasets['dev']

        :return: 返回的 :class:`~fastNLP.io.DataBundle`
        """
        if paths is None:
            paths = self.download()
        paths = check_loader_paths(paths)
        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
    
    def download(self) -> str:
        """
        自动下载该数据集

        :return: 下载后解压目录
        """
        raise NotImplementedError(f"{self.__class__} cannot download data automatically.")
    
    @staticmethod
    def _get_dataset_path(dataset_name):
        """
        传入dataset的名称，获取读取数据的目录。如果数据不存在，会尝试自动下载并缓存

        :param str dataset_name: 数据集的名称
        :return: str, 数据集的目录地址。直接到该目录下读取相应的数据即可。
        """
        
        default_cache_path = get_cache_path()
        url = _get_dataset_url(dataset_name)
        output_dir = cached_path(url_or_filename=url, cache_dir=default_cache_path, name='dataset')
        
        return output_dir
