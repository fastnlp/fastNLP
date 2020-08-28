r"""
Loader用于读取数据，并将内容读取到 :class:`~fastNLP.DataSet` 或者 :class:`~fastNLP.io.DataBundle` 中。所有的Loader都支持以下的
三个方法： ``__init__`` ， ``_load`` , ``loads`` . 其中 ``__init__(...)`` 用于申明读取参数，以及说明该Loader支持的数据格式，
读取后 :class:`~fastNLP.DataSet` 中的 `field` ; ``_load(path)`` 方法传入文件路径读取单个文件，并返回 :class:`~fastNLP.DataSet` ;
``load(paths)`` 用于读取文件夹下的文件，并返回 :class:`~fastNLP.io.DataBundle` 类型的对象 , load()方法支持以下几种类型的参数:

0.传入None
    将尝试自动下载数据集并缓存。但不是所有的数据都可以直接下载。

1.传入一个文件的 path
    返回的 `data_bundle` 包含一个名为 `train` 的 dataset ,可以通过 ``data_bundle.get_dataset('train')`` 获取

2.传入一个文件夹目录
    将读取的是这个文件夹下文件名中包含 `train` , `test` , `dev` 的文件，其它文件会被忽略。假设某个目录下的文件为::

        |
        +-train.txt
        +-dev.txt
        +-test.txt
        +-other.txt

    在 Loader().load('/path/to/dir') 返回的 `data_bundle` 中可以用 ``data_bundle.get_dataset('train')`` ,
    ``data_bundle.get_dataset('dev')`` ,
    ``data_bundle.get_dataset('test')`` 获取对应的 `dataset` ，其中 `other.txt` 的内容会被忽略。假设某个目录下的文件为::

        |
        +-train.txt
        +-dev.txt

    在 Loader().load('/path/to/dir') 返回的 `data_bundle` 中可以用 ``data_bundle.get_dataset('train')`` ,
    ``data_bundle.get_dataset('dev')`` 获取对应的 dataset。

3.传入一个字典
    字典的的 key 为 `dataset` 的名称，value 是该 `dataset` 的文件路径::

        paths = {'train':'/path/to/train', 'dev': '/path/to/dev', 'test':'/path/to/test'}
    
    在 Loader().load(paths)  返回的 `data_bundle` 中可以用 ``data_bundle.get_dataset('train')`` , ``data_bundle.get_dataset('dev')`` ,
    ``data_bundle.get_dataset('test')`` 来获取对应的 `dataset`

fastNLP 目前提供了如下的 Loader



"""

__all__ = [
    'Loader',
    
    'CLSBaseLoader',
    'YelpFullLoader',
    'YelpPolarityLoader',
    'AGsNewsLoader',
    'DBPediaLoader',
    'IMDBLoader',
    'SSTLoader',
    'SST2Loader',
    "ChnSentiCorpLoader",
    "THUCNewsLoader",
    "WeiboSenti100kLoader",
    
    'ConllLoader',
    'Conll2003Loader',
    'Conll2003NERLoader',
    'OntoNotesNERLoader',
    'CTBLoader',
    "MsraNERLoader",
    "PeopleDailyNERLoader",
    "WeiboNERLoader",
    
    'CSVLoader',
    'JsonLoader',
    
    'CWSLoader',
    
    'MNLILoader',
    "QuoraLoader",
    "SNLILoader",
    "QNLILoader",
    "RTELoader",
    "CNXNLILoader",
    "BQCorpusLoader",
    "LCQMCLoader",
    
    "CoReferenceLoader",

    "CMRC2018Loader"
]
from .classification import CLSBaseLoader, YelpFullLoader, YelpPolarityLoader, AGsNewsLoader, IMDBLoader, \
    SSTLoader, SST2Loader, DBPediaLoader, \
    ChnSentiCorpLoader, THUCNewsLoader, WeiboSenti100kLoader
from .conll import ConllLoader, Conll2003Loader, Conll2003NERLoader, OntoNotesNERLoader, CTBLoader
from .conll import MsraNERLoader, PeopleDailyNERLoader, WeiboNERLoader
from .coreference import CoReferenceLoader
from .csv import CSVLoader
from .cws import CWSLoader
from .json import JsonLoader
from .loader import Loader
from .matching import MNLILoader, QuoraLoader, SNLILoader, QNLILoader, RTELoader, CNXNLILoader, BQCorpusLoader, \
    LCQMCLoader
from .qa import CMRC2018Loader

