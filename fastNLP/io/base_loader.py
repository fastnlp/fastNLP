__all__ = [
    "BaseLoader",
    'DataBundle',
    'DataSetLoader',
]

import _pickle as pickle
from typing import Union, Dict
import os
from ..core.dataset import DataSet
from ..core.vocabulary import Vocabulary


class BaseLoader(object):
    """
    各个 Loader 的基类，提供了 API 的参考。

    """
    
    def __init__(self):
        super(BaseLoader, self).__init__()
    
    @staticmethod
    def load_lines(data_path):
        """
        按行读取，舍弃每行两侧空白字符，返回list of str

        :param data_path: 读取数据的路径
        """
        with open(data_path, "r", encoding="utf=8") as f:
            text = f.readlines()
        return [line.strip() for line in text]
    
    @classmethod
    def load(cls, data_path):
        """
        先按行读取，去除一行两侧空白，再提取每行的字符。返回list of list of str
        
        :param data_path:
        """
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.readlines()
        return [[word for word in sent.strip()] for sent in text]
    
    @classmethod
    def load_with_cache(cls, data_path, cache_path):
        """缓存版的load
        """
        if os.path.isfile(cache_path) and os.path.getmtime(data_path) < os.path.getmtime(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            obj = cls.load(data_path)
            with open(cache_path, 'wb') as f:
                pickle.dump(obj, f)
            return obj


def _download_from_url(url, path):
    try:
        from tqdm.auto import tqdm
    except:
        from ..core.utils import _pseudo_tqdm as tqdm
    import requests

    """Download file"""
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
    chunk_size = 16 * 1024
    total_size = int(r.headers.get('Content-length', 0))
    with open(path, "wb") as file, \
            tqdm(total=total_size, unit='B', unit_scale=1, desc=path.split('/')[-1]) as t:
        for chunk in r.iter_content(chunk_size):
            if chunk:
                file.write(chunk)
                t.update(len(chunk))


def _uncompress(src, dst):
    import zipfile
    import gzip
    import tarfile
    import os

    def unzip(src, dst):
        with zipfile.ZipFile(src, 'r') as f:
            f.extractall(dst)

    def ungz(src, dst):
        with gzip.open(src, 'rb') as f, open(dst, 'wb') as uf:
            length = 16 * 1024  # 16KB
            buf = f.read(length)
            while buf:
                uf.write(buf)
                buf = f.read(length)

    def untar(src, dst):
        with tarfile.open(src, 'r:gz') as f:
            f.extractall(dst)

    fn, ext = os.path.splitext(src)
    _, ext_2 = os.path.splitext(fn)
    if ext == '.zip':
        unzip(src, dst)
    elif ext == '.gz' and ext_2 != '.tar':
        ungz(src, dst)
    elif (ext == '.gz' and ext_2 == '.tar') or ext_2 == '.tgz':
        untar(src, dst)
    else:
        raise ValueError('unsupported file {}'.format(src))


class DataBundle:
    """
    经过处理的数据信息，包括一系列数据集（比如：分开的训练集、验证集和测试集）以及各个field对应的vocabulary。该对象一般由fastNLP中各种
    DataSetLoader的load函数生成，可以通过以下的方法获取里面的内容

    Example::

    :param vocabs: 从名称(字符串)到 :class:`~fastNLP.Vocabulary` 类型的dict
    :param datasets: 从名称(字符串)到 :class:`~fastNLP.DataSet` 类型的dict
    """

    def __init__(self, vocabs: dict = None, datasets: dict = None):
        self.vocabs = vocabs or {}
        self.datasets = datasets or {}

    def set_vocab(self, vocab, field_name):
        """
        向DataBunlde中增加vocab

        :param Vocabulary vocab: 词表
        :param str field_name: 这个vocab对应的field名称
        :return:
        """
        assert isinstance(vocab, Vocabulary), "Only fastNLP.Vocabulary supports."
        self.vocabs[field_name] = vocab

    def set_dataset(self, dataset, name):
        """

        :param DataSet dataset: 传递给DataBundle的DataSet
        :param str name: dataset的名称
        :return:
        """
        self.datasets[name] = dataset

    def get_dataset(self, name:str):
        """
        获取名为name的dataset

        :param str name: dataset的名称，一般为'train', 'dev', 'test'
        :return: DataSet
        """
        return self.datasets[name]

    def get_vocab(self, field_name:str):
        """
        获取field名为field_name对应的vocab

        :param str field_name: 名称
        :return: Vocabulary
        """
        return self.vocabs[field_name]

    def set_input(self, *field_names, flag=True, use_1st_ins_infer_dim_type=True, ignore_miss_field=True):
        """
        将field_names中的field设置为input, 对data_bundle中所有的dataset执行该操作::

            data_bundle.set_input('words', 'seq_len')   # 将words和seq_len这两个field的input属性设置为True
            data_bundle.set_input('words', flag=False)  # 将words这个field的input属性设置为False

        :param str field_names: field的名称
        :param bool flag: 将field_name的input状态设置为flag
        :param bool use_1st_ins_infer_dim_type: 如果为True，将不会check该列是否所有数据都是同样的维度，同样的类型。将直接使用第一
            行的数据进行类型和维度推断本列的数据的类型和维度。
        :param bool ignore_miss_field: 当某个field名称在某个dataset不存在时，如果为True，则直接忽略; 如果为False，则报错
        """
        for field_name in field_names:
            for name, dataset in self.datasets.items():
                if not ignore_miss_field and not dataset.has_field(field_name):
                    raise KeyError(f"Field:{field_name} was not found in DataSet:{name}")
                if not dataset.has_field(field_name):
                    continue
                else:
                    dataset.set_input(field_name, flag=flag, use_1st_ins_infer_dim_type=use_1st_ins_infer_dim_type)

    def set_target(self, *field_names, flag=True, use_1st_ins_infer_dim_type=True, ignore_miss_field=True):
        """
        将field_names中的field设置为target, 对data_bundle中所有的dataset执行该操作::

            data_bundle.set_target('target', 'seq_len')   # 将words和target这两个field的input属性设置为True
            data_bundle.set_target('target', flag=False)  # 将target这个field的input属性设置为False

        :param str field_names: field的名称
        :param bool flag: 将field_name的target状态设置为flag
        :param bool use_1st_ins_infer_dim_type: 如果为True，将不会check该列是否所有数据都是同样的维度，同样的类型。将直接使用第一
            行的数据进行类型和维度推断本列的数据的类型和维度。
        :param bool ignore_miss_field: 当某个field名称在某个dataset不存在时，如果为True，则直接忽略; 如果为False，则报错
        """
        for field_name in field_names:
            for name, dataset in self.datasets.items():
                if not ignore_miss_field and not dataset.has_field(field_name):
                    raise KeyError(f"Field:{field_name} was not found in DataSet:{name}")
                if not dataset.has_field(field_name):
                    continue
                else:
                    dataset.set_target(field_name, flag=flag, use_1st_ins_infer_dim_type=use_1st_ins_infer_dim_type)

    def __repr__(self):
        _str = 'In total {} datasets:\n'.format(len(self.datasets))
        for name, dataset in self.datasets.items():
            _str += '\t{} has {} instances.\n'.format(name, len(dataset))
        _str += 'In total {} vocabs:\n'.format(len(self.vocabs))
        for name, vocab in self.vocabs.items():
            _str += '\t{} has {} entries.\n'.format(name, len(vocab))
        return _str


class DataSetLoader:
    """
    别名：:class:`fastNLP.io.DataSetLoader` :class:`fastNLP.io.dataset_loader.DataSetLoader`

    定义了各种 DataSetLoader 所需的API 接口，开发者应该继承它实现各种的 DataSetLoader。

    开发者至少应该编写如下内容:

    - _load 函数：从一个数据文件中读取数据到一个 :class:`~fastNLP.DataSet`
    - load 函数（可以使用基类的方法）：从一个或多个数据文件中读取数据到一个或多个 :class:`~fastNLP.DataSet`
    - process 函数：一个或多个从数据文件中读取数据，并处理成可以训练的一个或多个 :class:`~fastNLP.DataSet`

    **process 函数中可以 调用load 函数或 _load 函数**

    """
    URL = ''
    DATA_DIR = ''

    ROOT_DIR = '.fastnlp/datasets/'
    UNCOMPRESS = True

    def _download(self, url: str, pdir: str, uncompress=True) -> str:
        """

        从 ``url`` 下载数据到 ``path``， 如果 ``uncompress`` 为 ``True`` ，自动解压。

        :param url: 下载的网站
        :param pdir: 下载到的目录
        :param uncompress:  是否自动解压缩
        :return: 数据的存放路径
        """
        fn = os.path.basename(url)
        path = os.path.join(pdir, fn)
        """check data exists"""
        if not os.path.exists(path):
            os.makedirs(pdir, exist_ok=True)
            _download_from_url(url, path)
        if uncompress:
            dst = os.path.join(pdir, 'data')
            if not os.path.exists(dst):
                _uncompress(path, dst)
            return dst
        return path

    def download(self):
        return self._download(
            self.URL,
            os.path.join(self.ROOT_DIR, self.DATA_DIR),
            uncompress=self.UNCOMPRESS)

    def load(self, paths: Union[str, Dict[str, str]]) -> Union[DataSet, Dict[str, DataSet]]:
        """
        从指定一个或多个路径中的文件中读取数据，返回一个或多个数据集 :class:`~fastNLP.DataSet` 。
        如果处理多个路径，传入的 dict 中的 key 与返回的 dict 中的 key 保存一致。

        :param Union[str, Dict[str, str]] paths: 文件路径
        :return: :class:`~fastNLP.DataSet` 类的对象或存储多个 :class:`~fastNLP.DataSet` 的字典
        """
        if isinstance(paths, str):
            return self._load(paths)
        return {name: self._load(path) for name, path in paths.items()}

    def _load(self, path: str) -> DataSet:
        """从指定路径的文件中读取数据,返回 :class:`~fastNLP.DataSet` 类型的对象

        :param str path: 文件路径
        :return: 一个 :class:`~fastNLP.DataSet` 类型的对象
        """
        raise NotImplementedError

    def process(self, paths: Union[str, Dict[str, str]], **options) -> DataBundle:
        """
        对于特定的任务和数据集，读取并处理数据，返回处理DataInfo类对象或字典。

        从指定一个或多个路径中的文件中读取数据，DataInfo对象中可以包含一个或多个数据集 。
        如果处理多个路径，传入的 dict 的 key 与返回DataInfo中的 dict 中的 key 保存一致。

        返回的 :class:`DataBundle` 对象有如下属性：

        - vocabs: 由从数据集中获取的词表组成的字典，每个词表
        - datasets: 一个dict，包含一系列 :class:`~fastNLP.DataSet` 类型的对象。其中 field 的命名参考 :mod:`~fastNLP.core.const`

        :param paths: 原始数据读取的路径
        :param options: 根据不同的任务和数据集，设计自己的参数
        :return: 返回一个 DataBundle
        """
        raise NotImplementedError
