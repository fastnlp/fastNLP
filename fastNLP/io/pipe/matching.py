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

'''
class BQCorpusPipe(_CLSPipe):
    """
    处理之后的DataSet有以下结构
    .. csv-table::
        :header: "raw_chars1", "raw_chars2", "chars1", "chars2", "target", "seq_len1", "seq_len2"

        "用微信都6年，微信没有微粒贷功能", "4。  号码来微粒贷", "[2, 3, ...]", "[4, 5, ...]", 0, 16, 12
        "..."

    其中chars1,chars2,seq_len1,seq_len2为input，target为target

    :param bool bigrams: 是否增加一列bigrams. bigrams的构成是['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]。如果
        设置为True，返回的DataSet将有一列名为bigrams, 且已经转换为了index并设置为input，对应的vocab可以通过
        data_bundle.get_vocab('bigrams')获取.
    :param bool trigrams: 是否增加一列trigrams. trigrams的构成是 ['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]
        。如果设置为True，返回的DataSet将有一列名为trigrams, 且已经转换为了index并设置为input，对应的vocab可以通过
        data_bundle.get_vocab('trigrams')获取.
    """

    def __init__(self, bigrams=False, trigrams=False):
        super().__init__()

        self.bigrams = bigrams
        self.trigrams = trigrams

    def _chracter_split(self, sent):
        return list(sent)

    def truncate_sentence(self, sentence):  # used for bert
        if (len(sentence) > 215):
            sentence = sentence[:215]
        return sentence

    def _tokenize(self, data_bundle, field_name=Const.INPUT, new_field_name=None):
        new_field_name = new_field_name or field_name
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self._chracter_split, field_name=field_name, new_field_name=new_field_name)
        return data_bundle

    def process(self, data_bundle: DataBundle):
        """
        可处理的DataSet应具备以下的field:

        .. csv-table::
            :header: "raw_chars1", "raw_chars2", "target"

            "不是邀请的如何贷款？", "我不是你们邀请的客人可以贷款吗？", "1"
            "如何满足微粒银行的审核", "建设银行有微粒贷的资格吗", "0"
            "...", "...", "..."

        :param data_bundle:
        :return:
        """
        # clean,lower

        # CWS(tokenize)
        data_bundle = self._tokenize(data_bundle, field_name='raw_chars1', new_field_name='chars1')
        data_bundle = self._tokenize(data_bundle, field_name='raw_chars2', new_field_name='chars2')

        input_field_names = [Const.CHAR_INPUT + '1', Const.CHAR_INPUT + '2']

        # n-grams
        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT + '1', new_field_name='bigrams1')
            input_field_names.append('bigrams1')
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT + '2', new_field_name='bigrams2')
            input_field_names.append('bigrams2')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT + '1', new_field_name='trigrams1')
            input_field_names.append('trigrams1')
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT + '2', new_field_name='trigrams2')
            input_field_names.append('trigrams2')

        # index
        data_bundle = _indexize(data_bundle, input_field_names=['chars1', 'chars2'])

        # add length
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(field_name='chars1', new_field_name="seq_len1")
            dataset.add_seq_len(field_name='chars2', new_field_name='seq_len2')

        input_fields = [Const.TARGET, 'seq_len1', 'seq_len2'] + input_field_names
        target_fields = [Const.TARGET]

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_for_bert(self, data_bundle: DataBundle, used_for_bert=True):
        """
        可处理的DataSet应具备以下的field:

        .. csv-table::
            :header: "raw_chars1", "raw_chars2", "target"

            "不是邀请的如何贷款？", "我不是你们邀请的客人可以贷款吗？", "1"
            "如何满足微粒银行的审核", "建设银行有微粒贷的资格吗", "0"
            "...", "...", "..."

        :param data_bundle:
        :return:
        """
        # clean,lower

        # CWS(tokenize)
        data_bundle = self._tokenize(data_bundle, field_name='raw_chars1', new_field_name='chars1')
        data_bundle = self._tokenize(data_bundle, field_name='raw_chars2', new_field_name='chars2')
        input_field_names = [Const.CHAR_INPUT + '1', Const.CHAR_INPUT + '2']

        # add length
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(field_name='chars1', new_field_name="seq_len1")
            dataset.add_seq_len(field_name='chars2', new_field_name='seq_len2')

        # (used for bert) cat, truncate(length after concatenation is supposed to be less than 430)
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.truncate_sentence, field_name='chars1')
            dataset.apply_field(self.truncate_sentence, field_name='chars2')

        # (used for bert)
        for name, dataset in data_bundle.datasets.items():
            dataset.apply(lambda ins: ins['chars1'] + ['[SEP]'] + ins['chars2'], new_field_name='chars')
        input_field_names = input_field_names + ['chars']

        # n-grams
        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT + '1', new_field_name='bigrams1')
            input_field_names.append('bigrams1')
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT + '2', new_field_name='bigrams2')
            input_field_names.append('bigrams2')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT + '1', new_field_name='trigrams1')
            input_field_names.append('trigrams1')
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT + '2', new_field_name='trigrams2')
            input_field_names.append('trigrams2')

        # index
        data_bundle = _indexize(data_bundle, input_field_names='chars')

        input_fields = [Const.TARGET, 'seq_len1', 'seq_len2'] + input_field_names
        target_fields = [Const.TARGET]

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths=None, used_for_bert=False):
        """
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.Loader` 的load函数。
        :return: DataBundle
        """
        data_loader = BQCorpusLoader()
        data_bundle = data_loader.load(paths)
        if not used_for_bert:
            data_bundle = self.process(data_bundle)
        else:
            data_bundle = self.process_for_bert(data_bundle)
        return data_bundle
'''

'''
class XNLIPipe(_CLSPipe):
    """
    处理之后的DataSet有以下结构
    .. csv-table::
        :header: "raw_chars1", "raw_chars2", "chars1", "chars2", "target", "seq_len1", "seq_len2"

        "从概念上看,奶油收入有两个基本方面产品和地理.", "产品和地理是什么使奶油抹霜工作.", "[88, 1059, 757, ...]", "[263, 319, 17, ...]", 1, 23, 16
        "..."

    其中chars1,chars2,seq_len1,seq_len2为input，target为target

    :param bool bigrams: 是否增加一列bigrams. bigrams的构成是['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]。如果
        设置为True，返回的DataSet将有一列名为bigrams, 且已经转换为了index并设置为input，对应的vocab可以通过
        data_bundle.get_vocab('bigrams')获取.
    :param bool trigrams: 是否增加一列trigrams. trigrams的构成是 ['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]
        。如果设置为True，返回的DataSet将有一列名为trigrams, 且已经转换为了index并设置为input，对应的vocab可以通过
        data_bundle.get_vocab('trigrams')获取.
    """

    def __init__(self, bigrams=False, trigrams=False):
        super().__init__()

        self.bigrams = bigrams
        self.trigrams = trigrams

    def _chracter_split(self, sent):
        return list(sent)

    def _XNLI_character_split(self, sent):
        return list("".join(sent.split()))  # 把已经分好词的premise和hypo强制还原为character segmentation

    def truncate_sentence(self, sentence):  # used for bert
        if (len(sentence) > 215):
            sentence = sentence[:215]
        return sentence

    def _tokenize(self, data_bundle, field_name=Const.INPUT, new_field_name=None, split_func=_chracter_split):
        new_field_name = new_field_name or field_name
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(split_func, field_name=field_name, new_field_name=new_field_name)
        return data_bundle

    def process(self, data_bundle: DataBundle):
        """
        可处理的DataSet应具备以下的field:

        .. csv-table::
           :header: "raw_chars1", "raw_chars2", "target"
           "从概念上看,奶油收入有两个基本方面产品和地理.", "产品和地理是什么使奶油抹霜工作.", "1"
           ""...", "...", "..."

        """
        # 根据granularity设置tag
        tag_map = {'neutral': 0, 'entailment': 1, 'contradictory': 2, 'contradiction': 2}
        data_bundle = self._granularize(data_bundle=data_bundle, tag_map=tag_map)

        # clean,lower

        # CWS(tokenize)
        data_bundle = self._tokenize(data_bundle, field_name='raw_chars1', new_field_name='chars1',
                                     split_func=self._XNLI_character_split)
        data_bundle = self._tokenize(data_bundle, field_name='raw_chars2', new_field_name='chars2',
                                     split_func=self._XNLI_character_split)

        input_field_names = [Const.CHAR_INPUT + '1', Const.CHAR_INPUT + '2']

        # n-grams
        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT + '1', new_field_name='bigrams1')
            input_field_names.append('bigrams1')
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT + '2', new_field_name='bigrams2')
            input_field_names.append('bigrams2')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT + '1', new_field_name='trigrams1')
            input_field_names.append('trigrams1')
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT + '2', new_field_name='trigrams2')
            input_field_names.append('trigrams2')

        # index
        data_bundle = _indexize(data_bundle, input_field_names=['chars1', 'chars2'])

        # add length
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(field_name='chars1', new_field_name="seq_len1")
            dataset.add_seq_len(field_name='chars2', new_field_name='seq_len2')

        input_fields = [Const.TARGET, 'seq_len1', 'seq_len2'] + input_field_names
        target_fields = [Const.TARGET]

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_for_bert(self, data_bundle: DataBundle):
        """
        可处理的DataSet应具备以下的field:

        .. csv-table::
           :header: "raw_chars1", "raw_chars2", "target"
           "从概念上看,奶油收入有两个基本方面产品和地理.", "产品和地理是什么使奶油抹霜工作.", "1"
           ""...", "...", "..."

        """
        # 根据granularity设置tag
        tag_map = {'neutral': 0, 'entailment': 1, 'contradictory': 2, 'contradiction': 2}
        data_bundle = self._granularize(data_bundle=data_bundle, tag_map=tag_map)

        # clean,lower

        # CWS(tokenize)
        data_bundle = self._tokenize(data_bundle, field_name='raw_chars1', new_field_name='chars1',
                                     split_func=self._XNLI_character_split)
        data_bundle = self._tokenize(data_bundle, field_name='raw_chars2', new_field_name='chars2',
                                     split_func=self._XNLI_character_split)
        input_field_names = [Const.CHAR_INPUT + '1', Const.CHAR_INPUT + '2']

        # add length
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(field_name='chars1', new_field_name="seq_len1")
            dataset.add_seq_len(field_name='chars2', new_field_name='seq_len2')

        # (used for bert) cat, truncate(length after concatenation is supposed to be less than 430)
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.truncate_sentence, field_name='chars1')
            dataset.apply_field(self.truncate_sentence, field_name='chars2')

        # (used for bert)
        for name, dataset in data_bundle.datasets.items():
            dataset.apply(lambda ins: ins['chars1'] + ['[SEP]'] + ins['chars2'], new_field_name='chars')
        input_field_names = input_field_names + ['chars']

        # n-grams
        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT + '1', new_field_name='bigrams1')
            input_field_names.append('bigrams1')
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT + '2', new_field_name='bigrams2')
            input_field_names.append('bigrams2')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT + '1', new_field_name='trigrams1')
            input_field_names.append('trigrams1')
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT + '2', new_field_name='trigrams2')
            input_field_names.append('trigrams2')

        # index
        data_bundle = _indexize(data_bundle, input_field_names='chars')

        input_fields = [Const.TARGET, 'seq_len1', 'seq_len2'] + input_field_names
        target_fields = [Const.TARGET]

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths=None, used_for_bert=False):
        """
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.Loader` 的load函数。
        :return: DataBundle
        """
        data_loader = XNLILoader()
        data_bundle = data_loader.load(paths)
        if not used_for_bert:
            data_bundle = self.process(data_bundle)
        else:
            data_bundle = self.process_for_bert(data_bundle)
        return data_bundle
'''

'''
class LCQMCPipe(_CLSPipe):
    """
    处理之后的DataSet有以下结构
    .. csv-table::
        :header: "raw_chars1", "raw_chars2", "chars1", "chars2", "target", "seq_len1", "seq_len2"

        "大家觉得她好看吗", "大家觉得跑男好看吗？", "[45, 79, 526, ...]", "[[45, 79, 526, ...]", 1, 8, 10
        "..."

    其中

    :param bool bigrams: 是否增加一列bigrams. bigrams的构成是['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]。如果
        设置为True，返回的DataSet将有一列名为bigrams, 且已经转换为了index并设置为input，对应的vocab可以通过
        data_bundle.get_vocab('bigrams')获取.
    :param bool trigrams: 是否增加一列trigrams. trigrams的构成是 ['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]
        。如果设置为True，返回的DataSet将有一列名为trigrams, 且已经转换为了index并设置为input，对应的vocab可以通过
        data_bundle.get_vocab('trigrams')获取.
    """

    def __init__(self, bigrams=False, trigrams=False):
        super().__init__()

        self.bigrams = bigrams
        self.trigrams = trigrams

    def _chracter_split(self, sent):
        return list(sent)

    def truncate_sentence(self, sentence):  # used for bert
        if (len(sentence) > 215):
            sentence = sentence[:215]
        return sentence

    def _tokenize(self, data_bundle, field_name=Const.INPUT, new_field_name=None):
        new_field_name = new_field_name or field_name
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self._chracter_split, field_name=field_name, new_field_name=new_field_name)
        return data_bundle


    def process(self, data_bundle: DataBundle):
        """
        可以处理的DataSet因该具备以下的field

        .. csv-table::
            :header: "raw_chars1", "raw_chars2", "target"
            "喜欢打篮球的男生喜欢什么样的女生？", "爱打篮球的男生喜欢什么样的女生？", "1"
            "晚上睡觉带着耳机听音乐有什么害处吗？", "妇可以戴耳机听音乐吗?", "0"
            ""...", "...", "..."

        :param data_bundle:
        :return:
        """
        # clean,lower

        # CWS(tokenize)
        data_bundle = self._tokenize(data_bundle, field_name='raw_chars1', new_field_name='chars1')
        data_bundle = self._tokenize(data_bundle, field_name='raw_chars2', new_field_name='chars2')

        input_field_names = [Const.CHAR_INPUT + '1', Const.CHAR_INPUT + '2']

        # n-grams
        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT + '1', new_field_name='bigrams1')
            input_field_names.append('bigrams1')
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT + '2', new_field_name='bigrams2')
            input_field_names.append('bigrams2')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT + '1', new_field_name='trigrams1')
            input_field_names.append('trigrams1')
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT + '2', new_field_name='trigrams2')
            input_field_names.append('trigrams2')

        # index
        data_bundle = _indexize(data_bundle, input_field_names=['chars1', 'chars2'])

        # add length
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(field_name='chars1', new_field_name="seq_len1")
            dataset.add_seq_len(field_name='chars2', new_field_name='seq_len2')

        input_fields = [Const.TARGET, 'seq_len1', 'seq_len2'] + input_field_names
        target_fields = [Const.TARGET]

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_for_bert(self, data_bundle: DataBundle):
        """
        可以处理的DataSet因该具备以下的field

        .. csv-table::
            :header: "raw_chars1", "raw_chars2", "target"
            "喜欢打篮球的男生喜欢什么样的女生？", "爱打篮球的男生喜欢什么样的女生？", "1"
            "晚上睡觉带着耳机听音乐有什么害处吗？", "妇可以戴耳机听音乐吗?", "0"
            ""...", "...", "..."

        :param data_bundle:
        :return:
        """
        # clean,lower

        # CWS(tokenize)
        data_bundle = self._tokenize(data_bundle, field_name='raw_chars1', new_field_name='chars1')
        data_bundle = self._tokenize(data_bundle, field_name='raw_chars2', new_field_name='chars2')

        input_field_names = [Const.CHAR_INPUT + '1', Const.CHAR_INPUT + '2']

        # add length
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(field_name='chars1', new_field_name="seq_len1")
            dataset.add_seq_len(field_name='chars2', new_field_name='seq_len2')

        # (used for bert) cat, truncate(length after concatenation is supposed to be less than 430)
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.truncate_sentence, field_name='chars1')
            dataset.apply_field(self.truncate_sentence, field_name='chars2')

        # (used for bert)
        for name, dataset in data_bundle.datasets.items():
            dataset.apply(lambda ins: ins['chars1'] + ['[SEP]'] + ins['chars2'], new_field_name='chars')
        input_field_names = input_field_names + ['chars']

        # n-grams
        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT + '1', new_field_name='bigrams1')
            input_field_names.append('bigrams1')
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT + '2', new_field_name='bigrams2')
            input_field_names.append('bigrams2')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT + '1', new_field_name='trigrams1')
            input_field_names.append('trigrams1')
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT + '2', new_field_name='trigrams2')
            input_field_names.append('trigrams2')

        # index
        data_bundle = _indexize(data_bundle, input_field_names='chars')

        input_fields = [Const.TARGET, 'seq_len1', 'seq_len2'] + input_field_names
        target_fields = [Const.TARGET]

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths=None, used_for_bert=False):
        """
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.Loader` 的load函数。
        :return: DataBundle
        """
        data_loader = LCQMCLoader()
        data_bundle = data_loader.load(paths)
        if not (used_for_bert):
            data_bundle = self.process(data_bundle)
        else:
            data_bundle = self.process_for_bert(data_bundle)
        return data_bundle
'''
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


