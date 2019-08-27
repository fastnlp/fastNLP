"""undocumented"""

__all__ = [
    "Conll2003NERPipe",
    "Conll2003Pipe",
    "OntoNotesNERPipe",
    "MsraNERPipe",
    "PeopleDailyPipe",
    "WeiboNERPipe"
]

from .pipe import Pipe
from .utils import _add_chars_field
from .utils import _indexize, _add_words_field
from .utils import iob2, iob2bioes
from .. import DataBundle
from ..loader.conll import Conll2003NERLoader, OntoNotesNERLoader
from ..loader.conll import PeopleDailyNERLoader, WeiboNERLoader, MsraNERLoader, ConllLoader
from ...core.const import Const
from ...core.vocabulary import Vocabulary


class _NERPipe(Pipe):
    """
    NER任务的处理Pipe, 该Pipe会（1）复制raw_words列，并命名为words; (2）在words, target列建立词表
    (创建 :class:`fastNLP.Vocabulary` 对象，所以在返回的DataBundle中将有两个Vocabulary); (3）将words，target列根据相应的
    Vocabulary转换为index。

    raw_words列为List[str], 是未转换的原始数据; words列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有words, target, seq_len; 设置为target有target, seq_len。

    :param: str encoding_type: target列使用什么类型的encoding方式，支持bioes, bio两种。
    :param bool lower: 是否将words小写化后再建立词表，绝大多数情况都不需要设置为True。
    """
    
    def __init__(self, encoding_type: str = 'bio', lower: bool = False):
        if encoding_type == 'bio':
            self.convert_tag = iob2
        else:
            self.convert_tag = lambda words: iob2bioes(iob2(words))
        self.lower = lower
    
    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        支持的DataSet的field为

        .. csv-table::
           :header: "raw_words", "target"

           "[Nadim, Ladki]", "[B-PER, I-PER]"
           "[AL-AIN, United, Arab, ...]", "[B-LOC, B-LOC, I-LOC, ...]"
           "[...]", "[...]"

        :param DataBundle data_bundle: 传入的DataBundle中的DataSet必须包含raw_words和ner两个field，且两个field的内容均为List[str]。
            在传入DataBundle基础上原位修改。
        :return: DataBundle
        """
        # 转换tag
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.convert_tag, field_name=Const.TARGET, new_field_name=Const.TARGET)
        
        _add_words_field(data_bundle, lower=self.lower)
        
        # index
        _indexize(data_bundle)
        
        input_fields = [Const.TARGET, Const.INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET, Const.INPUT_LEN]
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
        
        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)
        
        return data_bundle


class Conll2003NERPipe(_NERPipe):
    """
    Conll2003的NER任务的处理Pipe, 该Pipe会（1）复制raw_words列，并命名为words; (2）在words, target列建立词表
    (创建 :class:`fastNLP.Vocabulary` 对象，所以在返回的DataBundle中将有两个Vocabulary); (3）将words，target列根据相应的
    Vocabulary转换为index。
    经过该Pipe过后，DataSet中的内容如下所示

    .. csv-table:: Following is a demo layout of DataSet returned by Conll2003Loader
       :header: "raw_words", "words", "target", "seq_len"

       "[Nadim, Ladki]", "[2, 3]", "[1, 2]", 2
       "[AL-AIN, United, Arab, ...]", "[4, 5, 6,...]", "[3, 4,...]", 6
       "[...]", "[...]", "[...]", .

    raw_words列为List[str], 是未转换的原始数据; words列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有words, target, seq_len; 设置为target有target。

    :param: str encoding_type: target列使用什么类型的encoding方式，支持bioes, bio两种。
    :param bool lower: 是否将words小写化后再建立词表，绝大多数情况都不需要设置为True。
    """
    
    def process_from_file(self, paths) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = Conll2003NERLoader().load(paths)
        data_bundle = self.process(data_bundle)
        
        return data_bundle


class Conll2003Pipe(Pipe):
    def __init__(self, chunk_encoding_type='bioes', ner_encoding_type='bioes', lower: bool = False):
        """
        经过该Pipe后，DataSet中的内容如下

        .. csv-table::
           :header: "raw_words", "words", "pos", "chunk", "ner", "seq_len"

           "[Nadim, Ladki]", "[2, 3]", "[0, 0]", "[1, 2]", "[1, 2]", 2
           "[AL-AIN, United, Arab, ...]", "[4, 5, 6,...]", "[1, 2...]", "[3, 4...]", "[3, 4...]", 6
           "[...]", "[...]", "[...]", "[...]", "[...]".

        其中words, seq_len是input; pos, chunk, ner, seq_len是target

        :param str chunk_encoding_type: 支持bioes, bio。
        :param str ner_encoding_type: 支持bioes, bio。
        :param bool lower: 是否将words列小写化后再建立词表
        """
        if chunk_encoding_type == 'bio':
            self.chunk_convert_tag = iob2
        else:
            self.chunk_convert_tag = lambda tags: iob2bioes(iob2(tags))
        if ner_encoding_type == 'bio':
            self.ner_convert_tag = iob2
        else:
            self.ner_convert_tag = lambda tags: iob2bioes(iob2(tags))
        self.lower = lower
    
    def process(self, data_bundle) -> DataBundle:
        """
        输入的DataSet应该类似于如下的形式

        .. csv-table::
           :header: "raw_words", "pos", "chunk", "ner"

           "[Nadim, Ladki]", "[NNP, NNP]", "[B-NP, I-NP]", "[B-PER, I-PER]"
           "[AL-AIN, United, Arab, ...]", "[NNP, NNP...]", "[B-NP, B-NP, ...]", "[B-LOC, B-LOC,...]"
           "[...]", "[...]", "[...]", "[...]".

        :param data_bundle:
        :return: 传入的DataBundle
        """
        # 转换tag
        for name, dataset in data_bundle.datasets.items():
            dataset.drop(lambda x: "-DOCSTART-" in x[Const.RAW_WORD])
            dataset.apply_field(self.chunk_convert_tag, field_name='chunk', new_field_name='chunk')
            dataset.apply_field(self.ner_convert_tag, field_name='ner', new_field_name='ner')
        
        _add_words_field(data_bundle, lower=self.lower)
        
        # index
        _indexize(data_bundle, input_field_names=Const.INPUT, target_field_names=['pos', 'ner'])
        # chunk中存在一些tag只在dev中出现，没在train中
        tgt_vocab = Vocabulary(unknown=None, padding=None)
        tgt_vocab.from_dataset(*data_bundle.datasets.values(), field_name='chunk')
        tgt_vocab.index_dataset(*data_bundle.datasets.values(), field_name='chunk')
        data_bundle.set_vocab(tgt_vocab, 'chunk')
        
        input_fields = [Const.INPUT, Const.INPUT_LEN]
        target_fields = ['pos', 'ner', 'chunk', Const.INPUT_LEN]
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
        
        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)
        
        return data_bundle
    
    def process_from_file(self, paths):
        """

        :param paths:
        :return:
        """
        data_bundle = ConllLoader(headers=['raw_words', 'pos', 'chunk', 'ner']).load(paths)
        return self.process(data_bundle)


class OntoNotesNERPipe(_NERPipe):
    """
    处理OntoNotes的NER数据，处理之后DataSet中的field情况为

    .. csv-table:: Following is a demo layout of DataSet returned by Conll2003Loader
       :header: "raw_words", "words", "target", "seq_len"

       "[Nadim, Ladki]", "[2, 3]", "[1, 2]", 2
       "[AL-AIN, United, Arab, ...]", "[4, 5, 6,...]", "[3, 4]", 6
       "[...]", "[...]", "[...]", .

    raw_words列为List[str], 是未转换的原始数据; words列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有words, target, seq_len; 设置为target有target。

    :param: str encoding_type: target列使用什么类型的encoding方式，支持bioes, bio两种。
    :param bool lower: 是否将words小写化后再建立词表，绝大多数情况都不需要设置为True。
    """
    
    def process_from_file(self, paths):
        data_bundle = OntoNotesNERLoader().load(paths)
        return self.process(data_bundle)


class _CNNERPipe(Pipe):
    """
    中文NER任务的处理Pipe, 该Pipe会（1）复制raw_chars列，并命名为chars; (2）在chars, target列建立词表
    (创建 :class:`fastNLP.Vocabulary` 对象，所以在返回的DataBundle中将有两个Vocabulary); (3）将chars，target列根据相应的
    Vocabulary转换为index。

    raw_chars列为List[str], 是未转换的原始数据; chars列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有chars, target, seq_len; 设置为target有target, seq_len。

    :param: str encoding_type: target列使用什么类型的encoding方式，支持bioes, bio两种。
    """
    
    def __init__(self, encoding_type: str = 'bio'):
        if encoding_type == 'bio':
            self.convert_tag = iob2
        else:
            self.convert_tag = lambda words: iob2bioes(iob2(words))
    
    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        支持的DataSet的field为

        .. csv-table::
           :header: "raw_chars", "target"

           "[相, 比, 之, 下,...]", "[O, O, O, O, ...]"
           "[青, 岛, 海, 牛, 队, 和, ...]", "[B-ORG, I-ORG, I-ORG, ...]"
           "[...]", "[...]"

        raw_chars列为List[str], 是未转换的原始数据; chars列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
        target。返回的DataSet中被设置为input有chars, target, seq_len; 设置为target有target。

        :param DataBundle data_bundle: 传入的DataBundle中的DataSet必须包含raw_words和ner两个field，且两个field的内容均为List[str]。
            在传入DataBundle基础上原位修改。
        :return: DataBundle
        """
        # 转换tag
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.convert_tag, field_name=Const.TARGET, new_field_name=Const.TARGET)
        
        _add_chars_field(data_bundle, lower=False)
        
        # index
        _indexize(data_bundle, input_field_names=Const.CHAR_INPUT, target_field_names=Const.TARGET)
        
        input_fields = [Const.TARGET, Const.CHAR_INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET, Const.INPUT_LEN]
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.CHAR_INPUT)
        
        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)
        
        return data_bundle


class MsraNERPipe(_CNNERPipe):
    """
    处理MSRA-NER的数据，处理之后的DataSet的field情况为

    .. csv-table::
       :header: "raw_chars", "chars", "target", "seq_len"

       "[相, 比, 之, 下,...]", "[2, 3, 4, 5, ...]", "[0, 0, 0, 0, ...]", 11
       "[青, 岛, 海, 牛, 队, 和, ...]", "[10, 21, ....]", "[1, 2, 3, ...]", 21
       "[...]", "[...]", "[...]", .

    raw_chars列为List[str], 是未转换的原始数据; chars列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有chars, target, seq_len; 设置为target有target。

    """
    
    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = MsraNERLoader().load(paths)
        return self.process(data_bundle)


class PeopleDailyPipe(_CNNERPipe):
    """
    处理people daily的ner的数据，处理之后的DataSet的field情况为

    .. csv-table::
       :header: "raw_chars", "chars", "target", "seq_len"

       "[相, 比, 之, 下,...]", "[2, 3, 4, 5, ...]", "[0, 0, 0, 0, ...]", 11
       "[青, 岛, 海, 牛, 队, 和, ...]", "[10, 21, ....]", "[1, 2, 3, ...]", 21
       "[...]", "[...]", "[...]", .

    raw_chars列为List[str], 是未转换的原始数据; chars列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有chars, target, seq_len; 设置为target有target。
    """
    
    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = PeopleDailyNERLoader().load(paths)
        return self.process(data_bundle)


class WeiboNERPipe(_CNNERPipe):
    """
    处理weibo的ner的数据，处理之后的DataSet的field情况为

    .. csv-table::
       :header: "raw_chars", "chars", "target", "seq_len"

       "[相, 比, 之, 下,...]", "[2, 3, 4, 5, ...]", "[0, 0, 0, 0, ...]", 11
       "[青, 岛, 海, 牛, 队, 和, ...]", "[10, 21, ....]", "[1, 2, 3, ...]", 21
       "[...]", "[...]", "[...]", .

    raw_chars列为List[str], 是未转换的原始数据; chars列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有chars, target, seq_len; 设置为target有target。

    :param: str encoding_type: target列使用什么类型的encoding方式，支持bioes, bio两种。
    """
    
    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = WeiboNERLoader().load(paths)
        return self.process(data_bundle)
