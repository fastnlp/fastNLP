__all__ = [
    "CWSLoader"
]

import glob
import os
import random
import shutil
import time

from .loader import Loader
from fastNLP.core.dataset import DataSet, Instance


class CWSLoader(Loader):
    r"""
    **Chinese word segmentor** 的 **Loader** 。如果您使用了该数据集，请引用以下的文章：Thomas Emerson, The Second International Chinese Word Segmentation Bakeoff,
    2005. 更多信息可以在 http://sighan.cs.uchicago.edu/bakeoff2005/ 查看。

    :class:`CWSLoader` 支持的数据格式为：一行一句话，不同词之间用空格隔开，例如::

        上海 浦东 开发 与 法制 建设 同步
        新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）
        ...

    读取的 :class:`~fastNLP.core.DataSet` 将具备以下的数据结构：

    .. csv-table::
       :header: "raw_words"

       "上海 浦东 开发 与 法制 建设 同步"
       "新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）"
       "..."

    :param dataset_name: data 的名称，支持 ``['pku', 'msra', 'cityu'(繁体), 'as'(繁体), None]``
    """

    def __init__(self, dataset_name: str = None):
        super().__init__()
        datanames = {'pku': 'cws-pku', 'msra': 'cws-msra', 'as': 'cws-as', 'cityu': 'cws-cityu'}
        if dataset_name in datanames:
            self.dataset_name = datanames[dataset_name]
        else:
            self.dataset_name = None

    def _load(self, path: str):
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    ds.append(Instance(raw_words=line))
        return ds

    def download(self, dev_ratio=0.1, re_download=False) -> str:
        r"""
        自动下载数据集。

        :param dev_ratio: 如果路径中没有验证集，从 train 划分多少作为 dev 的数据。 如果为 **0** ，则不划分 dev
        :param re_download: 是否重新下载数据，以重新切分数据。
        :return: 数据集的目录地址
        """
        if self.dataset_name is None:
            return ''
        data_dir = self._get_dataset_path(dataset_name=self.dataset_name)
        modify_time = 0
        for filepath in glob.glob(os.path.join(data_dir, '*')):
            modify_time = os.stat(filepath).st_mtime
            break
        if time.time() - modify_time > 1 and re_download:  # 通过这种比较丑陋的方式判断一下文件是否是才下载的
            shutil.rmtree(data_dir)
            data_dir = self._get_dataset_path(dataset_name=self.dataset_name)

        if not os.path.exists(os.path.join(data_dir, 'dev.txt')):
            if dev_ratio > 0:
                assert 0 < dev_ratio < 1, "dev_ratio should be in range (0,1)."
                try:
                    with open(os.path.join(data_dir, 'train.txt'), 'r', encoding='utf-8') as f, \
                            open(os.path.join(data_dir, 'middle_file.txt'), 'w', encoding='utf-8') as f1, \
                            open(os.path.join(data_dir, 'dev.txt'), 'w', encoding='utf-8') as f2:
                        for line in f:
                            if random.random() < dev_ratio:
                                f2.write(line)
                            else:
                                f1.write(line)
                    os.remove(os.path.join(data_dir, 'train.txt'))
                    os.renames(os.path.join(data_dir, 'middle_file.txt'), os.path.join(data_dir, 'train.txt'))
                finally:
                    if os.path.exists(os.path.join(data_dir, 'middle_file.txt')):
                        os.remove(os.path.join(data_dir, 'middle_file.txt'))

        return data_dir
