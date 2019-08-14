from .pipe import Pipe
from .. import DataBundle
from .utils import iob2, iob2bioes
from ... import Const
from ..loader.conll import Conll2003NERLoader, OntoNotesNERLoader
from .utils import _indexize, _add_words_field


class _NERPipe(Pipe):
    """
    NER任务的处理Pipe, 该Pipe会（1）复制raw_words列，并命名为words; (2）在words, target列建立词表
    (创建 :class:`fastNLP.Vocabulary` 对象，所以在返回的DataBundle中将有两个Vocabulary); (3）将words，target列根据相应的
    Vocabulary转换为index。

    raw_words列为List[str], 是未转换的原始数据; words列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有words, target, seq_len; 设置为target有target, seq_len。

    :param: str encoding_type: target列使用什么类型的encoding方式，支持bioes, bio两种。
    :param bool lower: 是否将words小写化后再建立词表，绝大多数情况都不需要设置为True。
    :param int target_pad_val: target的padding值，target这一列pad的位置值为target_pad_val。默认为-100。
    """
    def __init__(self, encoding_type:str='bio', lower:bool=False, target_pad_val=0):
        if  encoding_type == 'bio':
            self.convert_tag = iob2
        else:
            self.convert_tag = iob2bioes
        self.lower = lower
        self.target_pad_val = int(target_pad_val)

    def process(self, data_bundle:DataBundle)->DataBundle:
        """
        支持的DataSet的field为

        .. csv-table:: Following is a demo layout of DataSet returned by Conll2003Loader
           :header: "raw_words", "target"

           "[Nadim, Ladki]", "[B-PER, I-PER]"
           "[AL-AIN, United, Arab, ...]", "[B-LOC, B-LOC, I-LOC, ...]"
           "[...]", "[...]"


        :param DataBundle data_bundle: 传入的DataBundle中的DataSet必须包含raw_words和ner两个field，且两个field的内容均为List[str]。
            在传入DataBundle基础上原位修改。
        :return: DataBundle

        Example::

            data_bundle = Conll2003Loader().load('/path/to/conll2003/')
            data_bundle = Conll2003NERPipe().process(data_bundle)

            # 获取train
            tr_data = data_bundle.get_dataset('train')

            # 获取target这个field的词表
            target_vocab = data_bundle.get_vocab('target')
            # 获取words这个field的词表
            word_vocab = data_bundle.get_vocab('words')

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
            dataset.set_pad_val(Const.TARGET, self.target_pad_val)
            dataset.add_seq_len(Const.INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = Conll2003NERLoader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class Conll2003NERPipe(_NERPipe):
    """
    Conll2003的NER任务的处理Pipe, 该Pipe会（1）复制raw_words列，并命名为words; (2）在words, target列建立词表
    (创建 :class:`fastNLP.Vocabulary` 对象，所以在返回的DataBundle中将有两个Vocabulary); (3）将words，target列根据相应的
    Vocabulary转换为index。
    经过该Pipe过后，DataSet中的内容如下所示

    .. csv-table:: Following is a demo layout of DataSet returned by Conll2003Loader
       :header: "raw_words", "words", "target", "seq_len"

       "[Nadim, Ladki]", "[1, 2]", "[1, 2]", 2
       "[AL-AIN, United, Arab, ...]", "[3, 4, 5,...]", "[3, 4]", 10
       "[...]", "[...]", "[...]", .

    raw_words列为List[str], 是未转换的原始数据; words列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有words, target, seq_len; 设置为target有target。

    :param: str encoding_type: target列使用什么类型的encoding方式，支持bioes, bio两种。
    :param bool lower: 是否将words小写化后再建立词表，绝大多数情况都不需要设置为True。
    :param int target_pad_val: target的padding值，target这一列pad的位置值为target_pad_val。默认为-100。
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


class OntoNotesNERPipe(_NERPipe):
    """
    处理OntoNotes的NER数据，处理之后DataSet中的field情况为

    .. csv-table:: Following is a demo layout of DataSet returned by Conll2003Loader
       :header: "raw_words", "words", "target", "seq_len"

       "[Nadim, Ladki]", "[1, 2]", "[1, 2]", 2
       "[AL-AIN, United, Arab, ...]", "[3, 4, 5,...]", "[3, 4]", 6
       "[...]", "[...]", "[...]", .


    :param bool lower: 是否将words小写化后再建立词表，绝大多数情况都不需要设置为True。
    :param bool delete_unused_fields: 是否删除NER任务中用不到的field。
    :param int target_pad_val: target的padding值，target这一列pad的位置值为target_pad_val。默认为-100。
    """

    def process_from_file(self, paths):
        data_bundle = OntoNotesNERLoader().load(paths)
        return self.process(data_bundle)

