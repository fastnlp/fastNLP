r"""undocumented"""

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
    读取之后的DataSet中的field情况为

    .. csv-table::
       :header: "text", "summary", "label", "publication"

       ["I got new tires from them and... ","..."], ["The new tires...","..."], [0, 1], "cnndm"
       ["Don't waste your time.  We had two...","..."], ["Time is precious","..."], [1], "cnndm"
       ["..."], ["..."], [], "cnndm"

    """

    def __init__(self, fields=None):
        fields = fields or {"text": None, "summary": None, "label": None, "publication": None}
        super(ExtCNNDMLoader, self).__init__(fields=fields)

    def load(self, paths: Union[str, Dict[str, str]] = None):
        r"""
        从指定一个或多个路径中的文件中读取数据，返回 :class:`~fastNLP.io.DataBundle` 。

        读取的field根据ExtCNNDMLoader初始化时传入的headers决定。

        :param str paths: 传入一个目录, 将在该目录下寻找train.label.jsonl, dev.label.jsonl
            test.label.jsonl三个文件(该目录还应该需要有一个名字为vocab的文件，在 :class:`~fastNLP.io.ExtCNNDMPipe`
            当中需要用到)。

        :return: 返回 :class:`~fastNLP.io.DataBundle`
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
        如果你使用了这个数据，请引用

        https://arxiv.org/pdf/1506.03340.pdf
        :return:
        """
        output_dir = self._get_dataset_path('ext-cnndm')
        return output_dir
