__all__ = [
    "ExtCNNDMLoader"
]

import os
from typing import Union, Dict

from ..data_bundle import DataBundle
from ..utils import check_loader_paths
from .json import JsonLoader


class ExtCNNDMLoader(JsonLoader):
    r"""
    **CNN / Daily Mail** 数据集的 **Loader** ，用于 **extractive summarization task** 任务。
    如果你使用了这个数据，请引用 https://arxiv.org/pdf/1506.03340.pdf
    
    读取的 :class:`~fastNLP.core.DataSet` 将具备以下的数据结构：

    .. csv-table::
       :header: "text", "summary", "label", "publication"

       "['I got new tires from them and... ','...']", "['The new tires...','...']", "[0, 1]", "cnndm"
       "['Don't waste your time.  We had two...','...']", "['Time is precious','...']", "[1]", "cnndm"
       "['...']", "['...']", "[]", "cnndm"

    :param fields:
    """

    def __init__(self, fields=None):
        fields = fields or {"text": None, "summary": None, "label": None, "publication": None}
        super(ExtCNNDMLoader, self).__init__(fields=fields)

    def load(self, paths: Union[str, Dict[str, str]] = None):
        r"""
        从指定一个或多个路径中的文件中读取数据，返回 :class:`~fastNLP.io.DataBundle` 。

        读取的 field 根据 :class:`ExtCNNDMLoader` 初始化时传入的 ``fields`` 决定。

        :param paths: 传入一个目录, 将在该目录下寻找 ``train.label.jsonl`` , ``dev.label.jsonl`` , 
            ``test.label.jsonl`` 三个文件（该目录还应该需要有一个名字为 ``vocab`` 的文件，在 :class:`~fastNLP.io.pipe.ExtCNNDMPipe`
            当中需要用到）。

        :return: :class:`~fastNLP.io.DataBundle`
        """
        if paths is None:
            paths = self.download()
        paths = check_loader_paths(paths)
        if ('train' in paths) and ('test' not in paths):
            paths['test'] = paths['train']
            paths.pop('train')

        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle

    def download(self):
        r"""
        自动下载数据集。

        :return: 数据集目录地址
        """
        output_dir = self._get_dataset_path('ext-cnndm')
        return output_dir
