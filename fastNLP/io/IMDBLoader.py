import os
from fastNLP import DataSet, Instance
from fastNLP.io import DataSetLoader


class IMDBLoader(DataSetLoader):
    """
    读取IMDB数据集，DataSet包含fields：

        words: list(str) 需要分类的文本
        label: str 文本的标签

    数据来源：https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    """
    def __init__(self):
        super(IMDBLoader, self).__init__()

    def _load(self, data_path, batch_size=64, keep_case=False, max_sent_len=None):
        """
        :param data_path: 数据集路径，如 aclImdb/train
        :param batch_size:
        :param keep_case: 是否保持大小写，若为False，则全部小写
        :param max_sent_len: 最大截断长度
        :return:
        """
        ds = DataSet()
        for label in ['pos', 'neg']:
            path = os.path.join(data_path, label)
            fnames = os.listdir(path)
            for fname in fnames:
                with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                    sent = f.readline()
                    if not keep_case:
                        sent = sent.lower()
                    words = sent.split()
                    if max_sent_len is not None and len(words) > max_sent_len:
                        words = words[:max_sent_len]
                    ds.append(Instance(words=words, label=label))
        return ds
    

