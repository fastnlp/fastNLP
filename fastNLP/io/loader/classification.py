"""undocumented"""

__all__ = [
    "YelpLoader",
    "YelpFullLoader",
    "YelpPolarityLoader",
    "IMDBLoader",
    "SSTLoader",
    "SST2Loader",
]

import glob
import os
import random
import shutil
import time
import warnings

from .loader import Loader
from ...core.dataset import DataSet
from ...core.instance import Instance


class YelpLoader(Loader):
    """
    别名：:class:`fastNLP.io.YelpLoader` :class:`fastNLP.io.loader.YelpLoader`

    原始数据中内容应该为, 每一行为一个sample，第一个逗号之前为target，第一个逗号之后为文本内容。

    Example::
    
        "1","I got 'new' tires from the..."
        "1","Don't waste your time..."

    读取YelpFull, YelpPolarity的数据。可以通过xxx下载并预处理数据。
    读取的DataSet将具备以下的数据结构

    .. csv-table::
       :header: "raw_words", "target"

       "I got 'new' tires from them and... ", "1"
       "Don't waste your time.  We had two...", "1"
       "...", "..."

    """
    
    def __init__(self):
        super(YelpLoader, self).__init__()
    
    def _load(self, path: str = None):
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                sep_index = line.index(',')
                target = line[:sep_index]
                raw_words = line[sep_index + 1:]
                if target.startswith("\""):
                    target = target[1:]
                if target.endswith("\""):
                    target = target[:-1]
                if raw_words.endswith("\""):
                    raw_words = raw_words[:-1]
                if raw_words.startswith('"'):
                    raw_words = raw_words[1:]
                raw_words = raw_words.replace('""', '"')  # 替换双引号
                if raw_words:
                    ds.append(Instance(raw_words=raw_words, target=target))
        return ds


class YelpFullLoader(YelpLoader):
    def download(self, dev_ratio: float = 0.1, re_download: bool = False):
        """
        自动下载数据集，如果你使用了这个数据集，请引用以下的文章

        Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances
        in Neural Information Processing Systems 28 (NIPS 2015)

        根据dev_ratio的值随机将train中的数据取出一部分作为dev数据。下载完成后在output_dir中有train.csv, test.csv,
        dev.csv三个文件。

        :param float dev_ratio: 如果路径中没有dev集，从train划分多少作为dev的数据. 如果为0，则不划分dev。
        :param bool re_download: 是否重新下载数据，以重新切分数据。
        :return: str, 数据集的目录地址
        """
        
        dataset_name = 'yelp-review-full'
        data_dir = self._get_dataset_path(dataset_name=dataset_name)
        modify_time = 0
        for filepath in glob.glob(os.path.join(data_dir, '*')):
            modify_time = os.stat(filepath).st_mtime
            break
        if time.time() - modify_time > 1 and re_download:  # 通过这种比较丑陋的方式判断一下文件是否是才下载的
            shutil.rmtree(data_dir)
            data_dir = self._get_dataset_path(dataset_name=dataset_name)
        
        if not os.path.exists(os.path.join(data_dir, 'dev.csv')):
            if dev_ratio > 0:
                assert 0 < dev_ratio < 1, "dev_ratio should be in range (0,1)."
                try:
                    with open(os.path.join(data_dir, 'train.csv'), 'r', encoding='utf-8') as f, \
                            open(os.path.join(data_dir, 'middle_file.csv'), 'w', encoding='utf-8') as f1, \
                            open(os.path.join(data_dir, 'dev.csv'), 'w', encoding='utf-8') as f2:
                        for line in f:
                            if random.random() < dev_ratio:
                                f2.write(line)
                            else:
                                f1.write(line)
                    os.remove(os.path.join(data_dir, 'train.csv'))
                    os.renames(os.path.join(data_dir, 'middle_file.csv'), os.path.join(data_dir, 'train.csv'))
                finally:
                    if os.path.exists(os.path.join(data_dir, 'middle_file.csv')):
                        os.remove(os.path.join(data_dir, 'middle_file.csv'))
        
        return data_dir


class YelpPolarityLoader(YelpLoader):
    def download(self, dev_ratio: float = 0.1, re_download=False):
        """
        自动下载数据集，如果你使用了这个数据集，请引用以下的文章

        Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances
        in Neural Information Processing Systems 28 (NIPS 2015)

        根据dev_ratio的值随机将train中的数据取出一部分作为dev数据。下载完成后从train中切分dev_ratio这么多作为dev

        :param float dev_ratio: 如果路径中不存在dev.csv, 从train划分多少作为dev的数据。 如果为0，则不划分dev。
        :param bool re_download: 是否重新下载数据，以重新切分数据。
        :return: str, 数据集的目录地址
        """
        dataset_name = 'yelp-review-polarity'
        data_dir = self._get_dataset_path(dataset_name=dataset_name)
        modify_time = 0
        for filepath in glob.glob(os.path.join(data_dir, '*')):
            modify_time = os.stat(filepath).st_mtime
            break
        if time.time() - modify_time > 1 and re_download:  # 通过这种比较丑陋的方式判断一下文件是否是才下载的
            shutil.rmtree(data_dir)
            data_dir = self._get_dataset_path(dataset_name=dataset_name)
        
        if not os.path.exists(os.path.join(data_dir, 'dev.csv')):
            if dev_ratio > 0:
                assert 0 < dev_ratio < 1, "dev_ratio should be in range (0,1)."
                try:
                    with open(os.path.join(data_dir, 'train.csv'), 'r', encoding='utf-8') as f, \
                            open(os.path.join(data_dir, 'middle_file.csv'), 'w', encoding='utf-8') as f1, \
                            open(os.path.join(data_dir, 'dev.csv'), 'w', encoding='utf-8') as f2:
                        for line in f:
                            if random.random() < dev_ratio:
                                f2.write(line)
                            else:
                                f1.write(line)
                    os.remove(os.path.join(data_dir, 'train.csv'))
                    os.renames(os.path.join(data_dir, 'middle_file.csv'), os.path.join(data_dir, 'train.csv'))
                finally:
                    if os.path.exists(os.path.join(data_dir, 'middle_file.csv')):
                        os.remove(os.path.join(data_dir, 'middle_file.csv'))
        
        return data_dir


class IMDBLoader(Loader):
    """
    别名：:class:`fastNLP.io.IMDBLoader` :class:`fastNLP.io.loader.IMDBLoader`

    IMDBLoader读取后的数据将具有以下两列内容: raw_words: str, 需要分类的文本; target: str, 文本的标签
    DataSet具备以下的结构:

    .. csv-table::
       :header: "raw_words", "target"

       "Bromwell High is a cartoon ... ", "pos"
       "Story of a man who has ...", "neg"
       "...", "..."

    """
    
    def __init__(self):
        super(IMDBLoader, self).__init__()
    
    def _load(self, path: str):
        dataset = DataSet()
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                target = parts[0]
                words = parts[1]
                if words:
                    dataset.append(Instance(raw_words=words, target=target))
        
        if len(dataset) == 0:
            raise RuntimeError(f"{path} has no valid data.")
        
        return dataset
    
    def download(self, dev_ratio: float = 0.1, re_download=False):
        """
        自动下载数据集，如果你使用了这个数据集，请引用以下的文章

        http://www.aclweb.org/anthology/P11-1015

        根据dev_ratio的值随机将train中的数据取出一部分作为dev数据。下载完成后从train中切分0.1作为dev

        :param float dev_ratio: 如果路径中没有dev.txt。从train划分多少作为dev的数据. 如果为0，则不划分dev
        :param bool re_download: 是否重新下载数据，以重新切分数据。
        :return: str, 数据集的目录地址
        """
        dataset_name = 'aclImdb'
        data_dir = self._get_dataset_path(dataset_name=dataset_name)
        modify_time = 0
        for filepath in glob.glob(os.path.join(data_dir, '*')):
            modify_time = os.stat(filepath).st_mtime
            break
        if time.time() - modify_time > 1 and re_download:  # 通过这种比较丑陋的方式判断一下文件是否是才下载的
            shutil.rmtree(data_dir)
            data_dir = self._get_dataset_path(dataset_name=dataset_name)
        
        if not os.path.exists(os.path.join(data_dir, 'dev.csv')):
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


class SSTLoader(Loader):
    """
    别名：:class:`fastNLP.io.SSTLoader` :class:`fastNLP.io.loader.SSTLoader`

    读取之后的DataSet具有以下的结构

    .. csv-table:: 下面是使用SSTLoader读取的DataSet所具备的field
        :header: "raw_words"

        "(3 (2 It) (4 (4 (2 's) (4 (3 (2 a)..."
        "(4 (4 (2 Offers) (3 (3 (2 that) (3 (3 rare)..."
        "..."

    raw_words列是str。

    """
    
    def __init__(self):
        super().__init__()
    
    def _load(self, path: str):
        """
        从path读取SST文件

        :param str path: 文件路径
        :return: DataSet
        """
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    ds.append(Instance(raw_words=line))
        return ds
    
    def download(self):
        """
        自动下载数据集，如果你使用了这个数据集，请引用以下的文章

            https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf

        :return: str, 数据集的目录地址
        """
        output_dir = self._get_dataset_path(dataset_name='sst')
        return output_dir


class SST2Loader(Loader):
    """
    数据SST2的Loader
    读取之后DataSet将如下所示

    .. csv-table:: 下面是使用SSTLoader读取的DataSet所具备的field
        :header: "raw_words", "target"

        "it 's a charming and often affecting...", "1"
        "unflinchingly bleak and...", "0"
        "..."

    test的DataSet没有target列。
    """
    
    def __init__(self):
        super().__init__()
    
    def _load(self, path: str):
        """
        从path读取SST2文件

        :param str path: 数据路径
        :return: DataSet
        """
        ds = DataSet()
        
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()  # 跳过header
            if 'test' in os.path.split(path)[1]:
                warnings.warn("SST2's test file has no target.")
                for line in f:
                    line = line.strip()
                    if line:
                        sep_index = line.index('\t')
                        raw_words = line[sep_index + 1:]
                        if raw_words:
                            ds.append(Instance(raw_words=raw_words))
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_words = line[:-2]
                        target = line[-1]
                        if raw_words:
                            ds.append(Instance(raw_words=raw_words, target=target))
        return ds
    
    def download(self):
        """
        自动下载数据集，如果你使用了该数据集，请引用以下的文章

        https://nlp.stanford.edu/pubs/SocherBauerManningNg_ACL2013.pdf

        :return:
        """
        output_dir = self._get_dataset_path(dataset_name='sst-2')
        return output_dir
