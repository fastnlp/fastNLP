"""undocumented"""

__all__ = [
    "CoReferenceLoader",
]

from ...core.dataset import DataSet
from ..file_reader import _read_json
from ...core.instance import Instance
from ...core.const import Const
from .json import JsonLoader


class CoReferenceLoader(JsonLoader):
    """
    原始数据中内容应该为, 每一行为一个json对象,其中doc_key包含文章的种类信息，speakers包含每句话的说话者信息，cluster是指向现实中同一个事物的聚集，sentences是文本信息内容。

    Example::

       {"doc_key":"bc/cctv/00/cctv_001",
       "speakers":"[["Speaker1","Speaker1","Speaker1"],["Speaker1","Speaker1","Speaker1"]]",
       "clusters":"[[[2,3],[4,5]],[7,8],[18,20]]]",
       "sentences":[["I","have","an","apple"],["It","is","good"]]
       }

    读取预处理好的Conll2012数据。

    """
    def __init__(self, fields=None, dropna=False):
        super().__init__(fields, dropna)
        self.fields = {"doc_key": Const.RAW_WORDS(0), "speakers": Const.RAW_WORDS(1), "clusters": Const.RAW_WORDS(2),
                       "sentences": Const.RAW_WORDS(3)}

    def _load(self, path):
        """
        加载数据
        :param path: 数据文件路径，文件为json

        :return:
        """
        dataset = DataSet()
        for idx, d in _read_json(path, fields=self.fields_list, dropna=self.dropna):
            if self.fields:
                ins = {self.fields[k]: v for k, v in d.items()}
            else:
                ins = d
            dataset.append(Instance(**ins))
        return dataset

    def download(self):
        """
        由于版权限制，不能提供自动下载功能。可参考

        https://www.aclweb.org/anthology/W12-4501

        :return:
        """
        raise RuntimeError("CoReference cannot be downloaded automatically.")
