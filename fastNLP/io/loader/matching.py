r"""undocumented"""

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
    r"""
    读取的数据格式为：

    Example::

        index	promptID	pairID	genre	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	label1	gold_label
        0	31193	31193n	government	( ( Conceptually ( cream skimming ) ) ...
        1	101457	101457e	telephone	( you ( ( know ( during ( ( ( the season ) and ) ( i guess ) ) )...
        ...

    读取MNLI任务的数据，读取之后的DataSet中包含以下的内容，words0是sentence1, words1是sentence2, target是gold_label, 测试集中没
    有target列。

    .. csv-table::
       :header: "raw_words1", "raw_words2", "target"

       "Conceptually cream ...", "Product and geography...", "neutral"
       "you know during the ...", "You lose the things to the...", "entailment"
       "...", "...", "..."

    """
    
    def __init__(self):
        super().__init__()
    
    def _load(self, path: str):
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()  # 跳过header
            if path.endswith("test_matched.tsv") or path.endswith('test_mismatched.tsv'):
                warnings.warn("MNLI's test file has no target.")
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        raw_words1 = parts[8]
                        raw_words2 = parts[9]
                        idx = int(parts[0])
                        if raw_words1 and raw_words2:
                            ds.append(Instance(raw_words1=raw_words1, raw_words2=raw_words2, index=idx))
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        raw_words1 = parts[8]
                        raw_words2 = parts[9]
                        target = parts[-1]
                        idx = int(parts[0])
                        if raw_words1 and raw_words2 and target:
                            ds.append(Instance(raw_words1=raw_words1, raw_words2=raw_words2, target=target, index=idx))
        return ds
    
    def load(self, paths: str = None):
        r"""

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
        r"""
        如果你使用了这个数据，请引用

        https://www.nyu.edu/projects/bowman/multinli/paper.pdf
        :return:
        """
        output_dir = self._get_dataset_path('mnli')
        return output_dir


class SNLILoader(JsonLoader):
    r"""
    文件每一行是一个sample，每一行都为一个json对象，其数据格式为：

    Example::

        {"annotator_labels": ["neutral", "entailment", "neutral", "neutral", "neutral"], "captionID": "4705552913.jpg#2",
         "gold_label": "neutral", "pairID": "4705552913.jpg#2r1n",
         "sentence1": "Two women are embracing while holding to go packages.",
         "sentence1_binary_parse": "( ( Two women ) ( ( are ( embracing ( while ( holding ( to ( go packages ) ) ) ) ) ) . ) )",
         "sentence1_parse": "(ROOT (S (NP (CD Two) (NNS women)) (VP (VBP are) (VP (VBG embracing) (SBAR (IN while) (S (NP (VBG holding)) (VP (TO to) (VP (VB go) (NP (NNS packages)))))))) (. .)))",
         "sentence2": "The sisters are hugging goodbye while holding to go packages after just eating lunch.",
         "sentence2_binary_parse": "( ( The sisters ) ( ( are ( ( hugging goodbye ) ( while ( holding ( to ( ( go packages ) ( after ( just ( eating lunch ) ) ) ) ) ) ) ) ) . ) )",
         "sentence2_parse": "(ROOT (S (NP (DT The) (NNS sisters)) (VP (VBP are) (VP (VBG hugging) (NP (UH goodbye)) (PP (IN while) (S (VP (VBG holding) (S (VP (TO to) (VP (VB go) (NP (NNS packages)) (PP (IN after) (S (ADVP (RB just)) (VP (VBG eating) (NP (NN lunch))))))))))))) (. .)))"
         }

    读取之后的DataSet中的field情况为

    .. csv-table:: 下面是使用SNLILoader加载的DataSet所具备的field
       :header: "target", "raw_words1", "raw_words2",

       "neutral ", "Two women are embracing while holding..", "The sisters are hugging goodbye..."
       "entailment", "Two women are embracing while holding...", "Two woman are holding packages."
       "...", "...", "..."

    """
    
    def __init__(self):
        super().__init__(fields={
            'sentence1': Const.RAW_WORDS(0),
            'sentence2': Const.RAW_WORDS(1),
            'gold_label': Const.TARGET,
        })
    
    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        r"""
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
        r"""
        如果您的文章使用了这份数据，请引用

        http://nlp.stanford.edu/pubs/snli_paper.pdf

        :return: str
        """
        return self._get_dataset_path('snli')


class QNLILoader(JsonLoader):
    r"""
    第一行为标题(具体内容会被忽略)，之后每一行是一个sample，由index、问题、句子和标签构成（以制表符分割），数据结构如下：

    Example::

        index	question	sentence	label
        0	What came into force after the new constitution was herald?	As of that day, the new constitution heralding the Second Republic came into force.	entailment

    QNLI数据集的Loader,
    加载的DataSet将具备以下的field, raw_words1是question, raw_words2是sentence, target是label

    .. csv-table::
        :header: "raw_words1", "raw_words2", "target"

        "What came into force after the new...", "As of that day...", "entailment"
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
        r"""
        如果您的实验使用到了该数据，请引用

        https://arxiv.org/pdf/1809.05053.pdf

        :return:
        """
        return self._get_dataset_path('qnli')


class RTELoader(Loader):
    r"""
    第一行为标题(具体内容会被忽略)，之后每一行是一个sample，由index、句子1、句子2和标签构成（以制表符分割），数据结构如下：

    Example::

        index	sentence1	sentence2	label
        0	Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, according to the Christopher Reeve Foundation.	Christopher Reeve had an accident.	not_entailment

    RTE数据的loader
    加载的DataSet将具备以下的field, raw_words1是sentence0，raw_words2是sentence1, target是label

    .. csv-table::
        :header: "raw_words1", "raw_words2", "target"

        "Dana Reeve, the widow of the actor...", "Christopher Reeve had an...", "not_entailment"
        "...","..."

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
        r"""
        如果您的实验使用到了该数据，请引用GLUE Benchmark

        https://openreview.net/pdf?id=rJ4km2R5t7

        :return:
        """
        return self._get_dataset_path('rte')


class QuoraLoader(Loader):
    r"""
    Quora matching任务的数据集Loader

    支持读取的文件中的内容，应该有以下的形式, 以制表符分隔，且前三列的内容必须是：第一列是label，第二列和第三列是句子

    Example::

        1	How do I get funding for my web based startup idea ?	How do I get seed funding pre product ?	327970
        0	Is honey a viable alternative to sugar for diabetics ?	How would you compare the United States ' euthanasia laws to Denmark ?	90348
        ...

    加载的DataSet将具备以下的field

    .. csv-table::
        :header: "raw_words1", "raw_words2", "target"

        "How do I get funding for my web based...", "How do I get seed funding...","1"
        "Is honey a viable alternative ...", "How would you compare the United...","0"
        "...","...","..."

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
        r"""
        由于版权限制，不能提供自动下载功能。可参考

        https://www.kaggle.com/c/quora-question-pairs/data

        :return:
        """
        raise RuntimeError("Quora cannot be downloaded automatically.")


class CNXNLILoader(Loader):
    r"""
    数据集简介：中文句对NLI（本为multi-lingual的数据集，但是这里只取了中文的数据集）。原句子已被MOSES tokenizer处理，这里我们将其还原并重新按字tokenize
    原始数据数据为：

    Example::

        premise	hypo	label
        我们 家里 有 一个 但 我 没 找到 我 可以 用 的 时间	我们 家里 有 一个 但 我 从来 没有 时间 使用 它 .	entailment

    dev和test中的数据为csv或json格式，包括十多个field，这里只取与以上三个field中的数据
    读取后的Dataset将具有以下数据结构：

    .. csv-table::
       :header: "raw_chars1", "raw_chars2", "target"
       
       "我们 家里 有 一个 但 我 没 找到 我 可以 用 的 时间", "我们 家里 有 一个 但 我 从来 没有 时间 使用 它 .", "0"
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
        r"""
        自动下载数据，该数据取自 https://arxiv.org/abs/1809.05053
        在 https://arxiv.org/pdf/1905.05526.pdf https://arxiv.org/pdf/1901.10125.pdf
        https://arxiv.org/pdf/1809.05053.pdf 有使用
        :return:
        """
        output_dir = self._get_dataset_path('cn-xnli')
        return output_dir


class BQCorpusLoader(Loader):
    r"""
    别名：
    数据集简介:句子对二分类任务（判断是否具有相同的语义）
    原始数据结构为：

    Example::

        sentence1,sentence2,label
        综合评分不足什么原因,综合评估的依据,0
        什么时候我能使用微粒贷,你就赶快给我开通就行了,0

    读取后的Dataset将具有以下数据结构：

    .. csv-table::
       :header: "raw_chars1", "raw_chars2", "target"
       
       "综合评分不足什么原因", "综合评估的依据", "0"
       "什么时候我能使用微粒贷", "你就赶快给我开通就行了", "0"
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
        r"""
        由于版权限制，不能提供自动下载功能。可参考

        https://github.com/ymcui/Chinese-BERT-wwm

        :return:
        """
        raise RuntimeError("BQCorpus cannot be downloaded automatically.")


class LCQMCLoader(Loader):
    r"""
    数据集简介：句对匹配（question matching）
    
    原始数据为：

    Example::

        喜欢打篮球的男生喜欢什么样的女生	爱打篮球的男生喜欢什么样的女生	1
        你帮我设计小说的封面吧	谁能帮我给小说设计个封面？	0

    
    读取后的Dataset将具有以下的数据结构
    
    .. csv-table::
       :header: "raw_chars1", "raw_chars2", "target"
       
       "喜欢打篮球的男生喜欢什么样的女生", "爱打篮球的男生喜欢什么样的女生", "1"
       "你帮我设计小说的封面吧", "妇可以戴耳机听音乐吗?", "0"
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
        r"""
        由于版权限制，不能提供自动下载功能。可参考

        https://github.com/ymcui/Chinese-BERT-wwm

        :return:
        """
        raise RuntimeError("LCQMC cannot be downloaded automatically.")


