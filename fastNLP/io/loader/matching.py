import warnings
from .loader import Loader
from .json import JsonLoader
from ...core.const import Const
from .. import DataBundle
import os
from typing import Union, Dict
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

    def _load(self, path:str):
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()  # 跳过header
            if path.endswith("test.tsv"):
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

    def load(self, paths:str=None):
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

        files = {'dev_matched':"dev_matched.tsv",
                 "dev_mismatched":"dev_mismatched.tsv",
                 "test_matched":"test_matched.tsv",
                 "test_mismatched":"test_mismatched.tsv",
                 "train":'train.tsv'}

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

    def load(self, paths: Union[str, Dict[str, str]]=None) -> DataBundle:
        """
        从指定一个或多个路径中的文件中读取数据，返回:class:`~fastNLP.io.DataBundle` 。

        读取的field根据ConllLoader初始化时传入的headers决定。

        :param str paths: 传入一个目录, 将在该目录下寻找snli_1.0_train.jsonl, snli_1.0_dev.jsonl
            和snli_1.0_test.jsonl三个文件。

        :return: 返回的:class:`~fastNLP.io.DataBundle`
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

        .. todo::
            补充

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

    def _load(self, path:str):
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

    def _load(self, path:str):
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
        raise RuntimeError("Quora cannot be downloaded automatically.")
