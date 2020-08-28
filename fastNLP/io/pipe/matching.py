r"""undocumented"""

__all__ = [
    "MatchingBertPipe",
    "RTEBertPipe",
    "SNLIBertPipe",
    "QuoraBertPipe",
    "QNLIBertPipe",
    "MNLIBertPipe",
    "CNXNLIBertPipe",
    "BQCorpusBertPipe",
    "LCQMCBertPipe",
    "MatchingPipe",
    "RTEPipe",
    "SNLIPipe",
    "QuoraPipe",
    "QNLIPipe",
    "MNLIPipe",
    "LCQMCPipe",
    "CNXNLIPipe",
    "BQCorpusPipe",
    "RenamePipe",
    "GranularizePipe",
    "MachingTruncatePipe",
]

import warnings

from .pipe import Pipe
from .utils import get_tokenizer
from ..data_bundle import DataBundle
from ..loader.matching import SNLILoader, MNLILoader, QNLILoader, RTELoader, QuoraLoader, BQCorpusLoader, CNXNLILoader, \
    LCQMCLoader
from ...core._logger import logger
from ...core.const import Const
from ...core.vocabulary import Vocabulary


class MatchingBertPipe(Pipe):
    r"""
    Matching任务的Bert pipe，输出的DataSet将包含以下的field

    .. csv-table::
       :header: "raw_words1", "raw_words2", "target", "words", "seq_len"

       "The new rights are...", "Everyone really likes..", 1,  "[2, 3, 4, 5, ...]", 10
       "This site includes a...", "The Government Executive...", 0, "[11, 12, 13,...]", 5
       "...", "...", ., "[...]", .

    words列是将raw_words1(即premise), raw_words2(即hypothesis)使用"[SEP]"链接起来转换为index的。
    words列被设置为input，target列被设置为target和input(设置为input以方便在forward函数中计算loss，
    如果不在forward函数中计算loss也不影响，fastNLP将根据forward函数的形参名进行传参).

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+------------+------------+--------+-------+---------+
        | field_names | raw_words1 | raw_words2 | target | words | seq_len |
        +-------------+------------+------------+--------+-------+---------+
        |   is_input  |   False    |   False    | False  |  True |   True  |
        |  is_target  |   False    |   False    |  True  | False |  False  |
        | ignore_type |            |            | False  | False |  False  |
        |  pad_value  |            |            |   0    |   0   |    0    |
        +-------------+------------+------------+--------+-------+---------+

    """
    
    def __init__(self, lower=False, tokenizer: str = 'raw'):
        r"""
        
        :param bool lower: 是否将word小写化。
        :param str tokenizer: 使用什么tokenizer来将句子切分为words. 支持spacy, raw两种。raw即使用空格拆分。
        """
        super().__init__()
        
        self.lower = bool(lower)
        self.tokenizer = get_tokenizer(tokenize_method=tokenizer)
    
    def _tokenize(self, data_bundle, field_names, new_field_names):
        r"""

        :param DataBundle data_bundle: DataBundle.
        :param list field_names: List[str], 需要tokenize的field名称
        :param list new_field_names: List[str], tokenize之后field的名称，与field_names一一对应。
        :return: 输入的DataBundle对象
        """
        for name, dataset in data_bundle.datasets.items():
            for field_name, new_field_name in zip(field_names, new_field_names):
                dataset.apply_field(lambda words: self.tokenizer(words), field_name=field_name,
                                    new_field_name=new_field_name)
        return data_bundle
    
    def process(self, data_bundle):
        r"""
        输入的data_bundle中的dataset需要具有以下结构：

        .. csv-table::
            :header: "raw_words1", "raw_words2", "target"

            "Dana Reeve, the widow of the actor...", "Christopher Reeve had an...", "not_entailment"
            "...","..."

        :param data_bundle:
        :return:
        """
        for dataset in data_bundle.datasets.values():
            if dataset.has_field(Const.TARGET):
                dataset.drop(lambda x: x[Const.TARGET] == '-')
        
        for name, dataset in data_bundle.datasets.items():
            dataset.copy_field(Const.RAW_WORDS(0), Const.INPUTS(0), )
            dataset.copy_field(Const.RAW_WORDS(1), Const.INPUTS(1), )
        
        if self.lower:
            for name, dataset in data_bundle.datasets.items():
                dataset[Const.INPUTS(0)].lower()
                dataset[Const.INPUTS(1)].lower()
        
        data_bundle = self._tokenize(data_bundle, [Const.INPUTS(0), Const.INPUTS(1)],
                                     [Const.INPUTS(0), Const.INPUTS(1)])
        
        # concat两个words
        def concat(ins):
            words0 = ins[Const.INPUTS(0)]
            words1 = ins[Const.INPUTS(1)]
            words = words0 + ['[SEP]'] + words1
            return words
        
        for name, dataset in data_bundle.datasets.items():
            dataset.apply(concat, new_field_name=Const.INPUT)
            dataset.delete_field(Const.INPUTS(0))
            dataset.delete_field(Const.INPUTS(1))
        
        word_vocab = Vocabulary()
        word_vocab.from_dataset(*[dataset for name, dataset in data_bundle.datasets.items() if 'train' in name],
                                field_name=Const.INPUT,
                                no_create_entry_dataset=[dataset for name, dataset in data_bundle.datasets.items() if
                                                         'train' not in name])
        word_vocab.index_dataset(*data_bundle.datasets.values(), field_name=Const.INPUT)
        
        target_vocab = Vocabulary(padding=None, unknown=None)
        target_vocab.from_dataset(*[ds for name, ds in data_bundle.iter_datasets() if 'train' in name],
                                  field_name=Const.TARGET,
                                  no_create_entry_dataset=[ds for name, ds in data_bundle.iter_datasets()
                                                           if ('train' not in name) and (ds.has_field(Const.TARGET))]
                                  )
        if len(target_vocab._no_create_word) > 0:
            warn_msg = f"There are {len(target_vocab._no_create_word)} target labels" \
                       f" in {[name for name in data_bundle.datasets.keys() if 'train' not in name]} " \
                       f"data set but not in train data set!."
            warnings.warn(warn_msg)
            logger.warning(warn_msg)
        
        has_target_datasets = [dataset for name, dataset in data_bundle.datasets.items() if
                               dataset.has_field(Const.TARGET)]
        target_vocab.index_dataset(*has_target_datasets, field_name=Const.TARGET)
        
        data_bundle.set_vocab(word_vocab, Const.INPUT)
        data_bundle.set_vocab(target_vocab, Const.TARGET)
        
        input_fields = [Const.INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET]
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
            dataset.set_input(*input_fields, flag=True)
            for fields in target_fields:
                if dataset.has_field(fields):
                    dataset.set_target(fields, flag=True)
        
        return data_bundle


class RTEBertPipe(MatchingBertPipe):
    def process_from_file(self, paths=None):
        data_bundle = RTELoader().load(paths)
        return self.process(data_bundle)


class SNLIBertPipe(MatchingBertPipe):
    def process_from_file(self, paths=None):
        data_bundle = SNLILoader().load(paths)
        return self.process(data_bundle)


class QuoraBertPipe(MatchingBertPipe):
    def process_from_file(self, paths):
        data_bundle = QuoraLoader().load(paths)
        return self.process(data_bundle)


class QNLIBertPipe(MatchingBertPipe):
    def process_from_file(self, paths=None):
        data_bundle = QNLILoader().load(paths)
        return self.process(data_bundle)


class MNLIBertPipe(MatchingBertPipe):
    def process_from_file(self, paths=None):
        data_bundle = MNLILoader().load(paths)
        return self.process(data_bundle)


class MatchingPipe(Pipe):
    r"""
    Matching任务的Pipe。输出的DataSet将包含以下的field

    .. csv-table::
       :header: "raw_words1", "raw_words2", "target", "words1", "words2", "seq_len1", "seq_len2"

       "The new rights are...", "Everyone really likes..", 1,  "[2, 3, 4, 5, ...]", "[10, 20, 6]", 10, 13
       "This site includes a...", "The Government Executive...", 0, "[11, 12, 13,...]", "[2, 7, ...]", 6, 7
       "...", "...", ., "[...]", "[...]", ., .

    words1是premise，words2是hypothesis。其中words1,words2,seq_len1,seq_len2被设置为input；target被设置为target
    和input(设置为input以方便在forward函数中计算loss，如果不在forward函数中计算loss也不影响，fastNLP将根据forward函数
    的形参名进行传参)。

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+------------+------------+--------+--------+--------+----------+----------+
        | field_names | raw_words1 | raw_words2 | target | words1 | words2 | seq_len1 | seq_len2 |
        +-------------+------------+------------+--------+--------+--------+----------+----------+
        |   is_input  |   False    |   False    | False  |  True  |  True  |   True   |   True   |
        |  is_target  |   False    |   False    |  True  | False  | False  |  False   |  False   |
        | ignore_type |            |            | False  | False  | False  |  False   |  False   |
        |  pad_value  |            |            |   0    |   0    |   0    |    0     |    0     |
        +-------------+------------+------------+--------+--------+--------+----------+----------+

    """
    
    def __init__(self, lower=False, tokenizer: str = 'raw'):
        r"""
        
        :param bool lower: 是否将所有raw_words转为小写。
        :param str tokenizer: 将原始数据tokenize的方式。支持spacy, raw. spacy是使用spacy切分，raw就是用空格切分。
        """
        super().__init__()
        
        self.lower = bool(lower)
        self.tokenizer = get_tokenizer(tokenize_method=tokenizer)
    
    def _tokenize(self, data_bundle, field_names, new_field_names):
        r"""

        :param ~fastNLP.DataBundle data_bundle: DataBundle.
        :param list field_names: List[str], 需要tokenize的field名称
        :param list new_field_names: List[str], tokenize之后field的名称，与field_names一一对应。
        :return: 输入的DataBundle对象
        """
        for name, dataset in data_bundle.datasets.items():
            for field_name, new_field_name in zip(field_names, new_field_names):
                dataset.apply_field(lambda words: self.tokenizer(words), field_name=field_name,
                                    new_field_name=new_field_name)
        return data_bundle
    
    def process(self, data_bundle):
        r"""
        接受的DataBundle中的DataSet应该具有以下的field, target列可以没有

        .. csv-table::
           :header: "raw_words1", "raw_words2", "target"

           "The new rights are...", "Everyone really likes..", "entailment"
           "This site includes a...", "The Government Executive...", "not_entailment"
           "...", "..."

        :param ~fastNLP.DataBundle data_bundle: 通过loader读取得到的data_bundle，里面包含了数据集的原始数据内容
        :return: data_bundle
        """
        data_bundle = self._tokenize(data_bundle, [Const.RAW_WORDS(0), Const.RAW_WORDS(1)],
                                     [Const.INPUTS(0), Const.INPUTS(1)])
        
        for dataset in data_bundle.datasets.values():
            if dataset.has_field(Const.TARGET):
                dataset.drop(lambda x: x[Const.TARGET] == '-')
        
        if self.lower:
            for name, dataset in data_bundle.datasets.items():
                dataset[Const.INPUTS(0)].lower()
                dataset[Const.INPUTS(1)].lower()
        
        word_vocab = Vocabulary()
        word_vocab.from_dataset(*[dataset for name, dataset in data_bundle.datasets.items() if 'train' in name],
                                field_name=[Const.INPUTS(0), Const.INPUTS(1)],
                                no_create_entry_dataset=[dataset for name, dataset in data_bundle.datasets.items() if
                                                         'train' not in name])
        word_vocab.index_dataset(*data_bundle.datasets.values(), field_name=[Const.INPUTS(0), Const.INPUTS(1)])
        
        target_vocab = Vocabulary(padding=None, unknown=None)
        target_vocab.from_dataset(*[ds for name, ds in data_bundle.iter_datasets() if 'train' in name],
                                  field_name=Const.TARGET,
                                  no_create_entry_dataset=[ds for name, ds in data_bundle.iter_datasets()
                                                           if ('train' not in name) and (ds.has_field(Const.TARGET))]
                                  )
        if len(target_vocab._no_create_word) > 0:
            warn_msg = f"There are {len(target_vocab._no_create_word)} target labels" \
                       f" in {[name for name in data_bundle.datasets.keys() if 'train' not in name]} " \
                       f"data set but not in train data set!."
            warnings.warn(warn_msg)
            logger.warning(warn_msg)
        
        has_target_datasets = [dataset for name, dataset in data_bundle.datasets.items() if
                               dataset.has_field(Const.TARGET)]
        target_vocab.index_dataset(*has_target_datasets, field_name=Const.TARGET)
        
        data_bundle.set_vocab(word_vocab, Const.INPUTS(0))
        data_bundle.set_vocab(target_vocab, Const.TARGET)
        
        input_fields = [Const.INPUTS(0), Const.INPUTS(1), Const.INPUT_LENS(0), Const.INPUT_LENS(1)]
        target_fields = [Const.TARGET]
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUTS(0), Const.INPUT_LENS(0))
            dataset.add_seq_len(Const.INPUTS(1), Const.INPUT_LENS(1))
            dataset.set_input(*input_fields, flag=True)
            for fields in target_fields:
                if dataset.has_field(fields):
                    dataset.set_target(fields, flag=True)
        
        return data_bundle


class RTEPipe(MatchingPipe):
    def process_from_file(self, paths=None):
        data_bundle = RTELoader().load(paths)
        return self.process(data_bundle)


class SNLIPipe(MatchingPipe):
    def process_from_file(self, paths=None):
        data_bundle = SNLILoader().load(paths)
        return self.process(data_bundle)


class QuoraPipe(MatchingPipe):
    def process_from_file(self, paths):
        data_bundle = QuoraLoader().load(paths)
        return self.process(data_bundle)


class QNLIPipe(MatchingPipe):
    def process_from_file(self, paths=None):
        data_bundle = QNLILoader().load(paths)
        return self.process(data_bundle)


class MNLIPipe(MatchingPipe):
    def process_from_file(self, paths=None):
        data_bundle = MNLILoader().load(paths)
        return self.process(data_bundle)


class LCQMCPipe(MatchingPipe):
    def __init__(self, tokenizer='cn=char'):
        super().__init__(tokenizer=tokenizer)

    def process_from_file(self, paths=None):
        data_bundle = LCQMCLoader().load(paths)
        data_bundle = RenamePipe().process(data_bundle)
        data_bundle = self.process(data_bundle)
        data_bundle = RenamePipe().process(data_bundle)
        return data_bundle


class CNXNLIPipe(MatchingPipe):
    def __init__(self, tokenizer='cn-char'):
        super().__init__(tokenizer=tokenizer)

    def process_from_file(self, paths=None):
        data_bundle = CNXNLILoader().load(paths)
        data_bundle = GranularizePipe(task='XNLI').process(data_bundle)
        data_bundle = RenamePipe().process(data_bundle)  # 使中文数据的field
        data_bundle = self.process(data_bundle)
        data_bundle = RenamePipe().process(data_bundle)
        return data_bundle


class BQCorpusPipe(MatchingPipe):
    def __init__(self, tokenizer='cn-char'):
        super().__init__(tokenizer=tokenizer)

    def process_from_file(self, paths=None):
        data_bundle = BQCorpusLoader().load(paths)
        data_bundle = RenamePipe().process(data_bundle)
        data_bundle = self.process(data_bundle)
        data_bundle = RenamePipe().process(data_bundle)
        return data_bundle


class RenamePipe(Pipe):
    def __init__(self, task='cn-nli'):
        super().__init__()
        self.task = task
    
    def process(self, data_bundle: DataBundle):  # rename field name for Chinese Matching dataset
        if (self.task == 'cn-nli'):
            for name, dataset in data_bundle.datasets.items():
                if (dataset.has_field(Const.RAW_CHARS(0))):
                    dataset.rename_field(Const.RAW_CHARS(0), Const.RAW_WORDS(0))  # RAW_CHARS->RAW_WORDS
                    dataset.rename_field(Const.RAW_CHARS(1), Const.RAW_WORDS(1))
                elif (dataset.has_field(Const.INPUTS(0))):
                    dataset.rename_field(Const.INPUTS(0), Const.CHAR_INPUTS(0))  # WORDS->CHARS
                    dataset.rename_field(Const.INPUTS(1), Const.CHAR_INPUTS(1))
                    dataset.rename_field(Const.RAW_WORDS(0), Const.RAW_CHARS(0))
                    dataset.rename_field(Const.RAW_WORDS(1), Const.RAW_CHARS(1))
                else:
                    raise RuntimeError(
                        "field name of dataset is not qualified. It should have ether RAW_CHARS or WORDS")
        elif (self.task == 'cn-nli-bert'):
            for name, dataset in data_bundle.datasets.items():
                if (dataset.has_field(Const.RAW_CHARS(0))):
                    dataset.rename_field(Const.RAW_CHARS(0), Const.RAW_WORDS(0))  # RAW_CHARS->RAW_WORDS
                    dataset.rename_field(Const.RAW_CHARS(1), Const.RAW_WORDS(1))
                elif (dataset.has_field(Const.RAW_WORDS(0))):
                    dataset.rename_field(Const.RAW_WORDS(0), Const.RAW_CHARS(0))
                    dataset.rename_field(Const.RAW_WORDS(1), Const.RAW_CHARS(1))
                    dataset.rename_field(Const.INPUT, Const.CHAR_INPUT)
                else:
                    raise RuntimeError(
                        "field name of dataset is not qualified. It should have ether RAW_CHARS or RAW_WORDS"
                    )
        else:
            raise RuntimeError(
                "Only support task='cn-nli' or 'cn-nli-bert'"
            )
        
        return data_bundle


class GranularizePipe(Pipe):
    def __init__(self, task=None):
        super().__init__()
        self.task = task
    
    def _granularize(self, data_bundle, tag_map):
        r"""
        该函数对data_bundle中'target'列中的内容进行转换。

        :param data_bundle:
        :param dict tag_map: 将target列中的tag做以下的映射，比如{"0":0, "1":0, "3":1, "4":1}, 则会删除target为"2"的instance，
            且将"1"认为是第0类。
        :return: 传入的data_bundle
        """
        for name in list(data_bundle.datasets.keys()):
            dataset = data_bundle.get_dataset(name)
            dataset.apply_field(lambda target: tag_map.get(target, -100), field_name=Const.TARGET,
                                new_field_name=Const.TARGET)
            dataset.drop(lambda ins: ins[Const.TARGET] == -100)
            data_bundle.set_dataset(dataset, name)
        return data_bundle
    
    def process(self, data_bundle: DataBundle):
        task_tag_dict = {
            'XNLI': {'neutral': 0, 'entailment': 1, 'contradictory': 2, 'contradiction': 2}
        }
        if self.task in task_tag_dict:
            data_bundle = self._granularize(data_bundle=data_bundle, tag_map=task_tag_dict[self.task])
        else:
            raise RuntimeError(f"Only support {task_tag_dict.keys()} task_tag_map.")
        return data_bundle


class MachingTruncatePipe(Pipe):  # truncate sentence for bert, modify seq_len
    def __init__(self):
        super().__init__()
    
    def process(self, data_bundle: DataBundle):
        for name, dataset in data_bundle.datasets.items():
            pass
        return None


class LCQMCBertPipe(MatchingBertPipe):
    def __init__(self, tokenizer='cn=char'):
        super().__init__(tokenizer=tokenizer)

    def process_from_file(self, paths=None):
        data_bundle = LCQMCLoader().load(paths)
        data_bundle = RenamePipe(task='cn-nli-bert').process(data_bundle)
        data_bundle = self.process(data_bundle)
        data_bundle = TruncateBertPipe(task='cn').process(data_bundle)
        data_bundle = RenamePipe(task='cn-nli-bert').process(data_bundle)
        return data_bundle


class BQCorpusBertPipe(MatchingBertPipe):
    def __init__(self, tokenizer='cn-char'):
        super().__init__(tokenizer=tokenizer)

    def process_from_file(self, paths=None):
        data_bundle = BQCorpusLoader().load(paths)
        data_bundle = RenamePipe(task='cn-nli-bert').process(data_bundle)
        data_bundle = self.process(data_bundle)
        data_bundle = TruncateBertPipe(task='cn').process(data_bundle)
        data_bundle = RenamePipe(task='cn-nli-bert').process(data_bundle)
        return data_bundle


class CNXNLIBertPipe(MatchingBertPipe):
    def __init__(self, tokenizer='cn-char'):
        super().__init__(tokenizer=tokenizer)

    def process_from_file(self, paths=None):
        data_bundle = CNXNLILoader().load(paths)
        data_bundle = GranularizePipe(task='XNLI').process(data_bundle)
        data_bundle = RenamePipe(task='cn-nli-bert').process(data_bundle)
        data_bundle = self.process(data_bundle)
        data_bundle = TruncateBertPipe(task='cn').process(data_bundle)
        data_bundle = RenamePipe(task='cn-nli-bert').process(data_bundle)
        return data_bundle


class TruncateBertPipe(Pipe):
    def __init__(self, task='cn'):
        super().__init__()
        self.task = task

    def _truncate(self, sentence_index:list, sep_index_vocab):
        # 根据[SEP]在vocab中的index，找到[SEP]在dataset的field['words']中的index
        sep_index_words = sentence_index.index(sep_index_vocab)
        words_before_sep = sentence_index[:sep_index_words]
        words_after_sep = sentence_index[sep_index_words:]  # 注意此部分包括了[SEP]
        if self.task == 'cn':
            # 中文任务将Instance['words']中在[SEP]前后的文本分别截至长度不超过250
            words_before_sep = words_before_sep[:250]
            words_after_sep = words_after_sep[:250]
        elif self.task == 'en':
            # 英文任务将Instance['words']中在[SEP]前后的文本分别截至长度不超过215
            words_before_sep = words_before_sep[:215]
            words_after_sep = words_after_sep[:215]
        else:
            raise RuntimeError("Only support 'cn' or 'en' task.")

        return words_before_sep + words_after_sep

    def process(self, data_bundle: DataBundle) -> DataBundle:
        for name in data_bundle.datasets.keys():
            dataset = data_bundle.get_dataset(name)
            sep_index_vocab = data_bundle.get_vocab('words').to_index('[SEP]')
            dataset.apply_field(lambda sent_index: self._truncate(sentence_index=sent_index, sep_index_vocab=sep_index_vocab), field_name='words', new_field_name='words')

            # truncate之后需要更新seq_len
            dataset.add_seq_len(field_name='words')
        return data_bundle

