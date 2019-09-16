"""undocumented"""

__all__ = [
    "MatchingBertPipe",
    "RTEBertPipe",
    "SNLIBertPipe",
    "QuoraBertPipe",
    "QNLIBertPipe",
    "MNLIBertPipe",
    "MatchingPipe",
    "RTEPipe",
    "SNLIPipe",
    "QuoraPipe",
    "QNLIPipe",
    "MNLIPipe",
    "XNLIPipe",
    "BQCorpusPipe",
    "LCQMCPipe"
]

from .pipe import Pipe
from .utils import get_tokenizer, _indexize
from ..loader.matching import SNLILoader, MNLILoader, QNLILoader, RTELoader, QuoraLoader, BQCorpusLoader, XNLILoader, LCQMCLoader
from ...core.const import Const
from ...core.vocabulary import Vocabulary
from .classification import _CLSPipe
from ..data_bundle import DataBundle

class MatchingBertPipe(Pipe):
    """
    Matching任务的Bert pipe，输出的DataSet将包含以下的field

    .. csv-table::
       :header: "raw_words1", "raw_words2", "words", "target", "seq_len"

       "The new rights are...", "Everyone really likes..",  "[2, 3, 4, 5, ...]", 1, 10
       "This site includes a...", "The Government Executive...", "[11, 12, 13,...]", 0, 5
       "...", "...", "[...]", ., .

    words列是将raw_words1(即premise), raw_words2(即hypothesis)使用"[SEP]"链接起来转换为index的。
    words列被设置为input，target列被设置为target和input(设置为input以方便在forward函数中计算loss，
    如果不在forward函数中计算loss也不影响，fastNLP将根据forward函数的形参名进行传参).

    :param bool lower: 是否将word小写化。
    :param str tokenizer: 使用什么tokenizer来将句子切分为words. 支持spacy, raw两种。raw即使用空格拆分。
    """
    
    def __init__(self, lower=False, tokenizer: str = 'raw'):
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
                dataset.apply_field(lambda words: self.tokenizer(words), field_name=field_name,
                                    new_field_name=new_field_name)
        return data_bundle
    
    def process(self, data_bundle):
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
        target_vocab.from_dataset(data_bundle.datasets['train'], field_name=Const.TARGET)
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
    """
    Matching任务的Pipe。输出的DataSet将包含以下的field

    .. csv-table::
       :header: "raw_words1", "raw_words2", "words1", "words2", "target", "seq_len1", "seq_len2"

       "The new rights are...", "Everyone really likes..",  "[2, 3, 4, 5, ...]", "[10, 20, 6]", 1, 10, 13
       "This site includes a...", "The Government Executive...", "[11, 12, 13,...]", "[2, 7, ...]", 0, 6, 7
       "...", "...", "[...]", "[...]", ., ., .

    words1是premise，words2是hypothesis。其中words1,words2,seq_len1,seq_len2被设置为input；target被设置为target
    和input(设置为input以方便在forward函数中计算loss，如果不在forward函数中计算loss也不影响，fastNLP将根据forward函数
    的形参名进行传参)。

    :param bool lower: 是否将所有raw_words转为小写。
    :param str tokenizer: 将原始数据tokenize的方式。支持spacy, raw. spacy是使用spacy切分，raw就是用空格切分。
    """
    
    def __init__(self, lower=False, tokenizer: str = 'raw'):
        super().__init__()
        
        self.lower = bool(lower)
        self.tokenizer = get_tokenizer(tokenize_method = tokenizer)
    
    def _tokenize(self, data_bundle, field_names, new_field_names):
        """

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
        """
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
        target_vocab.from_dataset(data_bundle.datasets['train'], field_name=Const.TARGET)
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
    def process_from_file(self, paths = None):
        data_bundle = LCQMCLoader().load(paths)
        data_bundle = RenamePipe().process(data_bundle)
        data_bundle = self.process(data_bundle)
        data_bundle = RenamePipe().process(data_bundle)
        return data_bundle

class XNLIPipe(MatchingPipe):
    def process_from_file(self, paths = None):
        data_bundle = XNLILoader().load(paths)
        data_bundle = GranularizePipe(task = 'XNLI').process(data_bundle)
        data_bundle = RenamePipe().process(data_bundle) #使中文数据的field
        data_bundle = self.process(data_bundle)
        data_bundle = RenamePipe().process(data_bundle)
        return data_bundle

class BQCorpusPipe(MatchingPipe):
    def process_from_file(self, paths = None):
        data_bundle = BQCorpusLoader().load(paths)
        data_bundle = RenamePipe().process(data_bundle)
        data_bundle = self.process(data_bundle)
        data_bundle = RenamePipe().process(data_bundle)
        return data_bundle

class RenamePipe(Pipe):
    def __init__(self):
        super().__init__()
    def process(self, data_bundle: DataBundle):  # rename field name for Chinese Matching dataset
        for name, dataset in data_bundle.datasets.items():
            if(dataset.has_field(Const.RAW_CHARS(0))):
                dataset.rename_field(Const.RAW_CHARS(0), Const.RAW_WORDS(0))  # RAW_CHARS->RAW_WORDS
                dataset.rename_field(Const.RAW_CHARS(1), Const.RAW_WORDS(1))
            elif(dataset.has_field(Const.INPUTS(0))):
                dataset.rename_field(Const.INPUTS(0), Const.CHAR_INPUTS(0)) #WORDS->CHARS
                dataset.rename_field(Const.INPUTS(1), Const.CHAR_INPUTS(1))
            else:
                raise RuntimeError(f"field name of dataset is not qualified. It should have ether RAW_CHARS or WORDS")
        return data_bundle

class GranularizePipe(Pipe):
    def __init__(self, task = None):
        super().__init__()
        self.task = task

    def _granularize(self, data_bundle, tag_map):
        """
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
            'XNLI':{'neutral': 0, 'entailment': 1, 'contradictory': 2, 'contradiction': 2}
        }
        if self.task in task_tag_dict:
            data_bundle = self._granularize(data_bundle=data_bundle, tag_map= task_tag_dict[self.task])
        else:
            raise RuntimeError(f"Only support {task_tag_dict.keys()} task_tag_map.")
        return data_bundle



def NGramPipe():#add n-gram features
    pass

def MachingTruncatePipe(): #truncate sentence for bert
    pass


