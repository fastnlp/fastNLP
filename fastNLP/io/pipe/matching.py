import math

from .pipe import Pipe
from .utils import get_tokenizer
from ...core import Const
from ...core import Vocabulary
from ..loader.matching import SNLILoader, MNLILoader, QNLILoader, RTELoader, QuoraLoader

class MatchingBertPipe(Pipe):
    """
    Matching任务的Bert pipe，输出的DataSet将包含以下的field

    .. csv-table::
       :header: "raw_words1", "raw_words2", "words", "target", "seq_len"

       "The new rights are...", "Everyone really likes..",  "[2, 3, 4, 5, ...]", 1, 10
       "This site includes a...", "The Government Executive...", "[11, 12, 13,...]", 0, 5
       "...", "...", "[...]", ., .

    words列是将raw_words1(即premise), raw_words2(即hypothesis)使用"[SEP]"链接起来转换为index的。
    words列被设置为input，target列被设置为target.

    :param bool lower: 是否将word小写化。
    :param str tokenizer: 使用什么tokenizer来将句子切分为words. 支持spacy, raw两种。raw即使用空格拆分。
    :param int max_concat_sent_length: 如果concat后的句子长度超过了该值，则合并后的句子将被截断到这个长度，截断时同时对premise
        和hypothesis按比例截断。
    """
    def __init__(self, lower=False, tokenizer:str='raw', max_concat_sent_length:int=480):
        super().__init__()

        self.lower = bool(lower)
        self.tokenizer = get_tokenizer(tokenizer=tokenizer)
        self.max_concat_sent_length = int(max_concat_sent_length)

    def _tokenize(self, data_bundle, field_names, new_field_names):
        """

        :param DataBundle data_bundle: DataBundle.
        :param list field_names: List[str], 需要tokenize的field名称
        :param list new_field_names: List[str], tokenize之后field的名称，与field_names一一对应。
        :return: 输入的DataBundle对象
        """
        for name, dataset in data_bundle.datasets.items():
            for field_name, new_field_name in zip(field_names, new_field_names):
                dataset.apply_field(lambda words:self.tokenizer(words), field_name=field_name,
                                    new_field_name=new_field_name)
        return data_bundle

    def process(self, data_bundle):
        for name, dataset in data_bundle.datasets.items():
            dataset.copy_field(Const.RAW_WORDS(0), Const.INPUTS(0))
            dataset.copy_field(Const.RAW_WORDS(1), Const.INPUTS(1))

        if self.lower:
            for name, dataset in data_bundle.datasets.items():
                dataset[Const.INPUTS(0)].lower()
                dataset[Const.INPUTS(1)].lower()

        data_bundle = self._tokenize(data_bundle, [Const.INPUTS(0), Const.INPUT(1)],
                                     [Const.INPUTS(0), Const.INPUTS(1)])

        # concat两个words
        def concat(ins):
            words0 = ins[Const.INPUTS(0)]
            words1 = ins[Const.INPUTS(1)]
            len0 = len(words0)
            len1 = len(words1)
            if len0 + len1 > self.max_concat_sent_length:
                ratio = self.max_concat_sent_length / (len0 + len1)
                len0 = math.floor(ratio * len0)
                len1 = math.floor(ratio * len1)
                words0 = words0[:len0]
                words1 = words1[:len1]

            words = words0 + ['[SEP]'] + words1
            return words
        for name, dataset in data_bundle.datasets.items():
            dataset.apply(concat, new_field_name=Const.INPUT)
            dataset.delete_field(Const.INPUTS(0))
            dataset.delete_field(Const.INPUTS(1))

        word_vocab = Vocabulary()
        word_vocab.from_dataset(data_bundle.datasets['train'], field_name=Const.INPUT,
                                no_create_entry_dataset=[dataset for name, dataset in data_bundle.datasets.items() if
                                                         name != 'train'])
        word_vocab.index_dataset(*data_bundle.datasets.values(), field_name=Const.INPUT)

        target_vocab = Vocabulary(padding=None, unknown=None)
        target_vocab.from_dataset(data_bundle.datasets['train'], field_name=Const.TARGET)
        has_target_datasets = []
        for name, dataset in data_bundle.datasets.items():
            if dataset.has_field(Const.TARGET):
                has_target_datasets.append(dataset)
        target_vocab.index_dataset(*has_target_datasets, field_name=Const.TARGET)

        data_bundle.set_vocab(word_vocab, Const.INPUT)
        data_bundle.set_vocab(target_vocab, Const.TARGET)

        input_fields = [Const.INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
            dataset.set_input(*input_fields, flag=True)
            dataset.set_target(*target_fields, flag=True)

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
    """
    Matching任务的Pipe。输出的DataSet将包含以下的field

    .. csv-table::
       :header: "raw_words1", "raw_words2", "words1", "words2", "target", "seq_len1", "seq_len2"

       "The new rights are...", "Everyone really likes..",  "[2, 3, 4, 5, ...]", "[10, 20, 6]", 1, 10, 13
       "This site includes a...", "The Government Executive...", "[11, 12, 13,...]", "[2, 7, ...]", 0, 6, 7
       "...", "...", "[...]", "[...]", ., ., .

    words1是premise，words2是hypothesis。其中words1,words2,seq_len1,seq_len2被设置为input；target被设置为target。

    :param bool lower: 是否将所有raw_words转为小写。
    :param str tokenizer: 将原始数据tokenize的方式。支持spacy, raw. spacy是使用spacy切分，raw就是用空格切分。
    """
    def __init__(self, lower=False, tokenizer:str='raw'):
        super().__init__()

        self.lower = bool(lower)
        self.tokenizer = get_tokenizer(tokenizer=tokenizer)

    def _tokenize(self, data_bundle, field_names, new_field_names):
        """

        :param DataBundle data_bundle: DataBundle.
        :param list field_names: List[str], 需要tokenize的field名称
        :param list new_field_names: List[str], tokenize之后field的名称，与field_names一一对应。
        :return: 输入的DataBundle对象
        """
        for name, dataset in data_bundle.datasets.items():
            for field_name, new_field_name in zip(field_names, new_field_names):
                dataset.apply_field(lambda words:self.tokenizer(words), field_name=field_name,
                                    new_field_name=new_field_name)
        return data_bundle

    def process(self, data_bundle):
        """
        接受的DataBundle中的DataSet应该具有以下的field, target列可以没有

        .. csv-table::
           :header: "raw_words1", "raw_words2", "target"

           "The new rights are...", "Everyone really likes..", "entailment"
           "This site includes a...", "The Government Executive...", "not_entailment"
           "...", "..."

        :param data_bundle:
        :return:
        """
        data_bundle = self._tokenize(data_bundle, [Const.RAW_WORDS(0), Const.RAW_WORDS(1)],
                                     [Const.INPUTS(0), Const.INPUTS(1)])

        if self.lower:
            for name, dataset in data_bundle.datasets.items():
                dataset[Const.INPUTS(0)].lower()
                dataset[Const.INPUTS(1)].lower()

        word_vocab = Vocabulary()
        word_vocab.from_dataset(data_bundle.datasets['train'], field_name=[Const.INPUTS(0), Const.INPUTS(1)],
                                no_create_entry_dataset=[dataset for name, dataset in data_bundle.datasets.items() if
                                                         name != 'train'])
        word_vocab.index_dataset(*data_bundle.datasets.values(), field_name=[Const.INPUTS(0), Const.INPUTS(1)])

        target_vocab = Vocabulary(padding=None, unknown=None)
        target_vocab.from_dataset(data_bundle.datasets['train'], field_name=Const.TARGET)
        has_target_datasets = []
        for name, dataset in data_bundle.datasets.items():
            if dataset.has_field(Const.TARGET):
                has_target_datasets.append(dataset)
        target_vocab.index_dataset(*has_target_datasets, field_name=Const.TARGET)

        data_bundle.set_vocab(word_vocab, Const.INPUTS(0))
        data_bundle.set_vocab(target_vocab, Const.TARGET)

        input_fields = [Const.INPUTS(0), Const.INPUTS(1), Const.INPUT_LEN(0), Const.INPUT_LEN(1)]
        target_fields = [Const.TARGET]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUTS(0), Const.INPUT_LEN(0))
            dataset.add_seq_len(Const.INPUTS(1), Const.INPUT_LEN(1))
            dataset.set_input(*input_fields, flag=True)
            dataset.set_target(*target_fields, flag=True)

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

