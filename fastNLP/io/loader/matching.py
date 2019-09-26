"""undocumented"""

__all__ = [
    "MNLILoader",
    "SNLILoader",
    "QNLILoader",
    "RTELoader",
    "QuoraLoader",
    "BQCorpusLoader",
    "CNXNLILoader",
    "LCQMCLoader"
]

import os
import warnings
from typing import Union, Dict

from .csv import CSVLoader
from .json import JsonLoader
from .loader import Loader
from .. import DataBundle
from ..utils import check_loader_paths
from ...core.const import Const
from ...core.dataset import DataSet
from ...core.instance import Instance


class MNLILoader(Loader):
    """
    读取MNLI任务的数据，读取之后的DataSet中包含以下的内容，words0是sentence1, words1是sentence2, target是gold_label, 测试集中没
    有target列。

    .. csv-table::
       :header: "raw_words1", "raw_words2", "target"

       "The new rights are...", "Everyone really likes..", "neutral"
       "This site includes a...", "The Government Executive...", "contradiction"
       "...", "...","."

    """
    
    def __init__(self):
        super().__init__()
    
    def _load(self, path: str):
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()  # 跳过header
            if path.endswith("test_matched.tsv") or path.endswith('test_mismatched.tsv'):
                warnings.warn("RTE's test file has no target.")
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        raw_words1 = parts[8]
                        raw_words2 = parts[9]
                        if raw_words1 and raw_words2:
                            ds.append(Instance(raw_words1=raw_words1, raw_words2=raw_words2))
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        raw_words1 = parts[8]
                        raw_words2 = parts[9]
                        target = parts[-1]
                        if raw_words1 and raw_words2 and target:
                            ds.append(Instance(raw_words1=raw_words1, raw_words2=raw_words2, target=target))
        return ds
    
    def load(self, paths: str = None):
        """

        :param str paths: 传入数据所在目录，会在该目录下寻找dev_matched.tsv, dev_mismatched.tsv, test_matched.tsv,
            test_mismatched.tsv, train.tsv文件夹
        :return: DataBundle
        """
        if paths:
            paths = os.path.abspath(os.path.expanduser(paths))
        else:
            paths = self.download()
        if not os.path.isdir(paths):
            raise NotADirectoryError(f"{paths} is not a valid directory.")
        
        files = {'dev_matched': "dev_matched.tsv",
                 "dev_mismatched": "dev_mismatched.tsv",
                 "test_matched": "test_matched.tsv",
                 "test_mismatched": "test_mismatched.tsv",
                 "train": 'train.tsv'}
        
        datasets = {}
        for name, filename in files.items():
            filepath = os.path.join(paths, filename)
            if not os.path.isfile(filepath):
                if 'test' not in name:
                    raise FileNotFoundError(f"{name} not found in directory {filepath}.")
            datasets[name] = self._load(filepath)
        
        data_bundle = DataBundle(datasets=datasets)
        
        return data_bundle
    
    def download(self):
        """
        如果你使用了这个数据，请引用

        https://www.nyu.edu/projects/bowman/multinli/paper.pdf
        :return:
        """
        output_dir = self._get_dataset_path('mnli')
        return output_dir


class SNLILoader(JsonLoader):
    """
    读取之后的DataSet中的field情况为

    .. csv-table:: 下面是使用SNLILoader加载的DataSet所具备的field
       :header: "raw_words1", "raw_words2", "target"

       "The new rights are...", "Everyone really likes..", "neutral"
       "This site includes a...", "The Government Executive...", "entailment"
       "...", "...", "."

    """
    
    def __init__(self):
        super().__init__(fields={
            'sentence1': Const.RAW_WORDS(0),
            'sentence2': Const.RAW_WORDS(1),
            'gold_label': Const.TARGET,
        })
    
    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        """
        从指定一个或多个路径中的文件中读取数据，返回 :class:`~fastNLP.io.DataBundle` 。

        读取的field根据Loader初始化时传入的field决定。

        :param str paths: 传入一个目录, 将在该目录下寻找snli_1.0_train.jsonl, snli_1.0_dev.jsonl
            和snli_1.0_test.jsonl三个文件。

        :return: 返回的 :class:`~fastNLP.io.DataBundle`
        """
        _paths = {}
        if paths is None:
            paths = self.download()
        if paths:
            if os.path.isdir(paths):
                if not os.path.isfile(os.path.join(paths, 'snli_1.0_train.jsonl')):
                    raise FileNotFoundError(f"snli_1.0_train.jsonl is not found in {paths}")
                _paths['train'] = os.path.join(paths, 'snli_1.0_train.jsonl')
                for filename in ['snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl']:
                    filepath = os.path.join(paths, filename)
                    _paths[filename.split('_')[-1].split('.')[0]] = filepath
                paths = _paths
            else:
                raise NotADirectoryError(f"{paths} is not a valid directory.")
        
        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
    
    def download(self):
        """
        如果您的文章使用了这份数据，请引用

        http://nlp.stanford.edu/pubs/snli_paper.pdf

        :return: str
        """
        return self._get_dataset_path('snli')


class QNLILoader(JsonLoader):
    """
    QNLI数据集的Loader,
    加载的DataSet将具备以下的field, raw_words1是question, raw_words2是sentence, target是label

    .. csv-table::
        :header: "raw_words1", "raw_words2", "target"

        "What came into force after the new...", "As of that day...", "entailment"
        "What is the first major...", "The most important tributaries", "not_entailment"
        "...","."

    test数据集没有target列

    """
    
    def __init__(self):
        super().__init__()
    
    def _load(self, path):
        ds = DataSet()
        
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()  # 跳过header
            if path.endswith("test.tsv"):
                warnings.warn("QNLI's test file has no target.")
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        raw_words1 = parts[1]
                        raw_words2 = parts[2]
                        if raw_words1 and raw_words2:
                            ds.append(Instance(raw_words1=raw_words1, raw_words2=raw_words2))
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        raw_words1 = parts[1]
                        raw_words2 = parts[2]
                        target = parts[-1]
                        if raw_words1 and raw_words2 and target:
                            ds.append(Instance(raw_words1=raw_words1, raw_words2=raw_words2, target=target))
        return ds
    
    def download(self):
        """
        如果您的实验使用到了该数据，请引用

        https://arxiv.org/pdf/1809.05053.pdf

        :return:
        """
        return self._get_dataset_path('qnli')


class RTELoader(Loader):
    """
    RTE数据的loader
    加载的DataSet将具备以下的field, raw_words1是sentence0，raw_words2是sentence1, target是label

    .. csv-table::
        :header: "raw_words1", "raw_words2", "target"

        "Dana Reeve, the widow of the actor...", "Christopher Reeve had an...", "not_entailment"
        "Yet, we now are discovering that...", "Bacteria is winning...", "entailment"
        "...","."

    test数据集没有target列
    """
    
    def __init__(self):
        super().__init__()
    
    def _load(self, path: str):
        ds = DataSet()
        
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()  # 跳过header
            if path.endswith("test.tsv"):
                warnings.warn("RTE's test file has no target.")
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        raw_words1 = parts[1]
                        raw_words2 = parts[2]
                        if raw_words1 and raw_words2:
                            ds.append(Instance(raw_words1=raw_words1, raw_words2=raw_words2))
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        raw_words1 = parts[1]
                        raw_words2 = parts[2]
                        target = parts[-1]
                        if raw_words1 and raw_words2 and target:
                            ds.append(Instance(raw_words1=raw_words1, raw_words2=raw_words2, target=target))
        return ds
    
    def download(self):
        """
        如果您的实验使用到了该数据，请引用GLUE Benchmark

        https://openreview.net/pdf?id=rJ4km2R5t7

        :return:
        """
        return self._get_dataset_path('rte')


class QuoraLoader(Loader):
    """
    Quora matching任务的数据集Loader

    支持读取的文件中的内容，应该有以下的形式, 以制表符分隔，且前三列的内容必须是：第一列是label，第二列和第三列是句子

    Example::

        1	How do I get funding for my web based startup idea ?	How do I get seed funding pre product ?	327970
        1	How can I stop my depression ?	What can I do to stop being depressed ?	339556
        ...

    加载的DataSet将具备以下的field

    .. csv-table::
        :header: "raw_words1", "raw_words2", "target"

        "What should I do to avoid...", "1"
        "How do I not sleep in a boring class...", "0"
        "...","."

    """
    
    def __init__(self):
        super().__init__()
    
    def _load(self, path: str):
        ds = DataSet()
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    raw_words1 = parts[1]
                    raw_words2 = parts[2]
                    target = parts[0]
                    if raw_words1 and raw_words2 and target:
                        ds.append(Instance(raw_words1=raw_words1, raw_words2=raw_words2, target=target))
        return ds
    
    def download(self):
        """
        由于版权限制，不能提供自动下载功能。可参考

        https://www.kaggle.com/c/quora-question-pairs/data

        :return:
        """
        raise RuntimeError("Quora cannot be downloaded automatically.")


class CNXNLILoader(Loader):
    """
    别名：
    数据集简介：中文句对NLI（本为multi-lingual的数据集，但是这里只取了中文的数据集）。原句子已被MOSES tokenizer处理，这里我们将其还原并重新按字tokenize
    原始数据为：
    train中的数据包括premise，hypo和label三个field
    dev和test中的数据为csv或json格式，包括十多个field，这里只取与以上三个field中的数据
    读取后的Dataset将具有以下数据结构：

    .. csv-table::
       :header: "raw_chars1", "raw_chars2", "target"
       
       "从概念上看,奶油收入有两个基本方面产品和地理.", "产品和地理是什么使奶油抹霜工作.", "1"
       "...", "...", "..."

    """

    def __init__(self):
        super(CNXNLILoader, self).__init__()

    def _load(self, path: str = None):
        ds_all = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            head_name_list = f.readline().strip().split('\t')
            sentence1_index = head_name_list.index('sentence1')
            sentence2_index = head_name_list.index('sentence2')
            gold_label_index = head_name_list.index('gold_label')
            language_index = head_name_list.index(('language'))

            for line in f:
                line = line.strip()
                raw_instance = line.split('\t')
                sentence1 = raw_instance[sentence1_index]
                sentence2 = raw_instance[sentence2_index]
                gold_label = raw_instance[gold_label_index]
                language = raw_instance[language_index]
                if sentence1:
                    ds_all.append(Instance(sentence1=sentence1, sentence2=sentence2, gold_label=gold_label, language=language))

        ds_zh = DataSet()
        for i in ds_all:
            if i['language'] == 'zh':
                ds_zh.append(Instance(raw_chars1=i['sentence1'], raw_chars2=i['sentence2'], target=i['gold_label']))

        return ds_zh

    def _load_train(self, path: str = None):
        ds = DataSet()

        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                raw_instance = line.strip().split('\t')
                premise = "".join(raw_instance[0].split())# 把已经分好词的premise和hypo强制还原为character segmentation
                hypo = "".join(raw_instance[1].split())
                label = "".join(raw_instance[-1].split())
                if premise:
                    ds.append(Instance(premise=premise, hypo=hypo, label=label))

        ds.rename_field('label', 'target')
        ds.rename_field('premise', 'raw_chars1')
        ds.rename_field('hypo', 'raw_chars2')
        ds.apply(lambda i: "".join(i['raw_chars1'].split()), new_field_name='raw_chars1')
        ds.apply(lambda i: "".join(i['raw_chars2'].split()), new_field_name='raw_chars2')
        return ds

    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        if paths is None:
            paths = self.download()
        paths = check_loader_paths(paths)
        datasets = {}
        for name, path in paths.items():
            if name == 'train':
                datasets[name] = self._load_train(path)
            else:
                datasets[name] = self._load(path)

        data_bundle = DataBundle(datasets=datasets)
        return data_bundle

    def download(self) -> str:
        """
        自动下载数据，该数据取自 https://arxiv.org/abs/1809.05053
        在 https://arxiv.org/pdf/1905.05526.pdf https://arxiv.org/pdf/1901.10125.pdf
        https://arxiv.org/pdf/1809.05053.pdf 有使用
        :return:
        """
        output_dir = self._get_dataset_path('cn-xnli')
        return output_dir


class BQCorpusLoader(Loader):
    """
    别名：
    数据集简介:句子对二分类任务（判断是否具有相同的语义）
    原始数据内容为：
    每行一个sample，第一个','之前为text1，第二个','之前为text2，第二个','之后为target
    第一行为sentence1 sentence2 label
    读取后的Dataset将具有以下数据结构：

    .. csv-table::
       :header: "raw_chars1", "raw_chars2", "target"
       
       "不是邀请的如何贷款？", "我不是你们邀请的客人可以贷款吗？", "1"
       "如何满足微粒银行的审核", "建设银行有微粒贷的资格吗", "0"
       "...", "...", "..."

    """

    def __init__(self):
        super(BQCorpusLoader, self).__init__()

    def _load(self, path: str = None):
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                line = line.strip()
                target = line[-1]
                sep_index = line.index(',')
                raw_chars1 = line[:sep_index]
                raw_chars2 = line[sep_index + 1:]

                if raw_chars1:
                    ds.append(Instance(raw_chars1=raw_chars1, raw_chars2=raw_chars2, target=target))
        return ds

    def download(self):
        """
        由于版权限制，不能提供自动下载功能。可参考

        https://github.com/ymcui/Chinese-BERT-wwm

        :return:
        """
        raise RuntimeError("BQCorpus cannot be downloaded automatically.")


class LCQMCLoader(Loader):
    r"""
    数据集简介：句对匹配（question matching）
    
    原始数据为：
    
    .. code-block:: text
    
        '喜欢打篮球的男生喜欢什么样的女生\t爱打篮球的男生喜欢什么样的女生\t1\n'
        '晚上睡觉带着耳机听音乐有什么害处吗？\t孕妇可以戴耳机听音乐吗?\t0\n'
    
    读取后的Dataset将具有以下的数据结构
    
    .. csv-table::
       :header: "raw_chars1", "raw_chars2", "target"
       
       "喜欢打篮球的男生喜欢什么样的女生？", "爱打篮球的男生喜欢什么样的女生？", "1"
       "晚上睡觉带着耳机听音乐有什么害处吗？", "妇可以戴耳机听音乐吗?", "0"
       "...", "...", "..."
    
    
    """

    def __init__(self):
        super(LCQMCLoader, self).__init__()

    def _load(self, path: str = None):
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line_segments = line.split('\t')
                assert len(line_segments) == 3

                target = line_segments[-1]

                raw_chars1 = line_segments[0]
                raw_chars2 = line_segments[1]

                if raw_chars1:
                    ds.append(Instance(raw_chars1=raw_chars1, raw_chars2=raw_chars2, target=target))
        return ds

    def download(self):
        """
        由于版权限制，不能提供自动下载功能。可参考

        https://github.com/ymcui/Chinese-BERT-wwm

        :return:
        """
        raise RuntimeError("LCQMC cannot be downloaded automatically.")


