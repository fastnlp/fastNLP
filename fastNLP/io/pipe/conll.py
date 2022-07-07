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
from fastNLP.io.data_bundle import DataBundle
from ..loader.conll import Conll2003NERLoader, OntoNotesNERLoader
from ..loader.conll import PeopleDailyNERLoader, WeiboNERLoader, MsraNERLoader, ConllLoader
# from ...core.const import Const
from ...core.vocabulary import Vocabulary


class _NERPipe(Pipe):
    r"""
    NER任务的处理Pipe, 该Pipe会（1）复制raw_words列，并命名为words; (2）在words, target列建立词表
    (创建 :class:`fastNLP.Vocabulary` 对象，所以在返回的DataBundle中将有两个Vocabulary); (3）将words，target列根据相应的
    Vocabulary转换为index。

    raw_words列为List[str], 是未转换的原始数据; words列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有words, target, seq_len; 设置为target有target, seq_len。
    """
    
    def __init__(self, encoding_type: str = 'bio', lower: bool = False, num_proc=0):
        r"""

        :param: str encoding_type: target列使用什么类型的encoding方式，支持bioes, bio两种。
        :param bool lower: 是否将words小写化后再建立词表，绝大多数情况都不需要设置为True。
        """
        if encoding_type == 'bio':
            self.convert_tag = iob2
        elif encoding_type == 'bioes':
            def func(words):
                return iob2bioes(iob2(words))
            # self.convert_tag = lambda words: iob2bioes(iob2(words))
            self.convert_tag = func
        else:
            raise ValueError("encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
        self.num_proc = num_proc
    
    def process(self, data_bundle: DataBundle) -> DataBundle:
        r"""
        支持的DataSet的field为

        .. csv-table::
           :header: "raw_words", "target"

           "[Nadim, Ladki]", "[B-PER, I-PER]"
           "[AL-AIN, United, Arab, ...]", "[B-LOC, B-LOC, I-LOC, ...]"
           "[...]", "[...]"

        :param ~fastNLP.DataBundle data_bundle: 传入的DataBundle中的DataSet必须包含raw_words和ner两个field，且两个field的内容均为List[str]在传入DataBundle基础上原位修改。
        :return: 处理后的 ``data_bundle``
        """
        # 转换tag
        for name, dataset in data_bundle.iter_datasets():
            dataset.apply_field(self.convert_tag, field_name='target', new_field_name='target', num_proc=self.num_proc)
        
        _add_words_field(data_bundle, lower=self.lower)
        
        # index
        _indexize(data_bundle)
        
        for name, dataset in data_bundle.iter_datasets():
            dataset.add_seq_len('words')

        return data_bundle


class Conll2003NERPipe(_NERPipe):
    r"""
    **Conll2003** 的 **NER** 任务的处理 **Pipe** ， 该Pipe会：
    
        1. 复制 ``raw_words`` 列，并命名为 ``words`` ；
        2. 在 ``words`` , ``target`` 列建立词表，即创建 :class:`~fastNLP.core.Vocabulary` 对象，所以在返回的
          :class:`~fastNLP.io.DataBundle` 中将有两个 ``Vocabulary`` ；
        3. 将 ``words`` , ``target`` 列根据相应的词表转换为 index。
    
    处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table:: Following is a demo layout of DataSet returned by Conll2003Loader
       :header: "raw_words", "target", "words", "seq_len"

       "[Nadim, Ladki]", "[1, 2]", "[2, 3]", 2
       "[AL-AIN, United, Arab, ...]", "[3, 4,...]", "[4, 5, 6,...]", 6
       "[...]", "[...]", "[...]", .

    ``raw_words`` 列为 :class:`List` [ :class:`str` ], 是未转换的原始数据； ``words`` 列为 :class:`List` [ :class:`int` ]，
    是转换为 index 的输入数据； ``target`` 列是 :class:`List` [ :class:`int` ] ，是转换为 index 的 target。返回的 :class:`~fastNLP.core.DataSet` 
    中被设置为 input 有 ``words`` , ``target``, ``seq_len``；target 有 ``target`` 。

    :param encoding_type: ``target`` 列使用什么类型的 encoding 方式，支持 ``['bioes', 'bio']`` 两种。
    :param lower: 是否将 ``words`` 小写化后再建立词表，绝大多数情况都不需要设置为 ``True`` 。
    :param num_proc: 处理数据时使用的进程数目。
    """
    
    def process_from_file(self, paths) -> DataBundle:
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load()`

        :param paths:
        :return:
        """
        # 读取数据
        data_bundle = Conll2003NERLoader().load(paths)
        data_bundle = self.process(data_bundle)
        
        return data_bundle


class Conll2003Pipe(Pipe):
    r"""
    处理 **Conll2003** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table::
       :header: "raw_words" , "pos", "chunk", "ner", "words", "seq_len"

       "[Nadim, Ladki]", "[0, 0]", "[1, 2]", "[1, 2]", "[2, 3]", 2
       "[AL-AIN, United, Arab, ...]", "[1, 2...]", "[3, 4...]", "[3, 4...]", "[4, 5, 6,...]", 6
       "[...]", "[...]", "[...]", "[...]", "[...]", .

    其中``words``, ``seq_len`` 是 input; ``pos``, ``chunk``, ``ner``, ``seq_len`` 是 target

    :param chunk_encoding_type: ``chunk`` 列使用什么类型的 encoding 方式，支持 ``['bioes', 'bio']`` 两种。
    :param ner_encoding_type: ``ner`` 列使用什么类型的 encoding 方式，支持 ``['bioes', 'bio']`` 两种。
    :param lower: 是否将 ``words`` 小写化后再建立词表，绝大多数情况都不需要设置为 ``True`` 。
    :param num_proc: 处理数据时使用的进程数目。
    """
    def __init__(self, chunk_encoding_type: str='bioes', ner_encoding_type: str='bioes', lower: bool = False, num_proc: int = 0):
        if chunk_encoding_type == 'bio':
            self.chunk_convert_tag = iob2
        elif chunk_encoding_type == 'bioes':
            def func1(tags):
                return iob2bioes(iob2(tags))
            # self.chunk_convert_tag = lambda tags: iob2bioes(iob2(tags))
            self.chunk_convert_tag = func1
        else:
            raise ValueError("chunk_encoding_type only supports `bio` and `bioes`.")
        if ner_encoding_type == 'bio':
            self.ner_convert_tag = iob2
        elif ner_encoding_type == 'bioes':
            def func2(tags):
                return iob2bioes(iob2(tags))
            # self.ner_convert_tag = lambda tags: iob2bioes(iob2(tags))
            self.ner_convert_tag = func2
        else:
            raise ValueError("ner_encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
        self.num_proc = num_proc
    
    def process(self, data_bundle) -> DataBundle:
        r"""
        输入的 `~fastNLP.core.DataSet` 应该类似于如下的形式：

        .. csv-table::
           :header: "raw_words", "pos", "chunk", "ner"

           "[Nadim, Ladki]", "[NNP, NNP]", "[B-NP, I-NP]", "[B-PER, I-PER]"
           "[AL-AIN, United, Arab, ...]", "[NNP, NNP...]", "[B-NP, B-NP, ...]", "[B-LOC, B-LOC,...]"
           "[...]", "[...]", "[...]", "[...]", .

        :param data_bundle:
        :return: 处理后的 ``data_bundle``
        """
        # 转换tag
        for name, dataset in data_bundle.datasets.items():
            dataset.drop(lambda x: "-DOCSTART-" in x['raw_words'])
            dataset.apply_field(self.chunk_convert_tag, field_name='chunk', new_field_name='chunk', num_proc=self.num_proc)
            dataset.apply_field(self.ner_convert_tag, field_name='ner', new_field_name='ner', num_proc=self.num_proc)
        
        _add_words_field(data_bundle, lower=self.lower)
        
        # index
        _indexize(data_bundle, input_field_names='words', target_field_names=['pos', 'ner'])
        # chunk中存在一些tag只在dev中出现，没在train中
        tgt_vocab = Vocabulary(unknown=None, padding=None)
        tgt_vocab.from_dataset(*data_bundle.datasets.values(), field_name='chunk')
        tgt_vocab.index_dataset(*data_bundle.datasets.values(), field_name='chunk')
        data_bundle.set_vocab(tgt_vocab, 'chunk')

        for name, dataset in data_bundle.iter_datasets():
            dataset.add_seq_len('words')

        return data_bundle
    
    def process_from_file(self, paths):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load()`

        :param paths:
        :return:
        """
        data_bundle = ConllLoader(headers=['raw_words', 'pos', 'chunk', 'ner']).load(paths)
        return self.process(data_bundle)


class OntoNotesNERPipe(_NERPipe):
    r"""
    处理 **OntoNotes** 的 **NER** 数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table::
       :header: "raw_words", "target", "words", "seq_len"

       "[Nadim, Ladki]", "[1, 2]", "[2, 3]", 2
       "[AL-AIN, United, Arab, ...]", "[3, 4]", "[4, 5, 6,...]", 6
       "[...]", "[...]", "[...]", .

    ``raw_words`` 列为 :class:`List` [ :class:`str` ], 是未转换的原始数据； ``words`` 列为 :class:`List` [ :class:`int` ]，
    是转换为 index 的输入数据； ``target`` 列是 :class:`List` [ :class:`int` ] ，是转换为 index 的 target。返回的 :class:`~fastNLP.core.DataSet` 
    中被设置为 input 有 ``words`` , ``target``, ``seq_len``；target 有 ``target`` 。

    """
    
    def process_from_file(self, paths):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load()`

        :param paths:
        :return:
        """
        data_bundle = OntoNotesNERLoader().load(paths)
        return self.process(data_bundle)


class _CNNERPipe(Pipe):
    r"""
    中文NER任务的处理Pipe, 该Pipe会（1）复制raw_chars列，并命名为chars; (2）在chars, target列建立词表
    (创建 :class:`fastNLP.Vocabulary` 对象，所以在返回的DataBundle中将有两个Vocabulary); (3）将chars，target列根据相应的
    Vocabulary转换为index。

    raw_chars列为List[str], 是未转换的原始数据; chars列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有chars, target, seq_len; 设置为target有target, seq_len。

    """
    
    def __init__(self, encoding_type: str = 'bio', bigrams=False, trigrams=False, num_proc: int = 0):
        r"""
        
        :param str encoding_type: target列使用什么类型的encoding方式，支持bioes, bio两种。
        :param bool bigrams: 是否增加一列bigrams. bigrams的构成是['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]。如果
            设置为True，返回的DataSet将有一列名为bigrams, 且已经转换为了index并设置为input，对应的vocab可以通过
            data_bundle.get_vocab('bigrams')获取.
        :param bool trigrams: 是否增加一列trigrams. trigrams的构成是 ['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]
            。如果设置为True，返回的DataSet将有一列名为trigrams, 且已经转换为了index并设置为input，对应的vocab可以通过
            data_bundle.get_vocab('trigrams')获取.
        """
        if encoding_type == 'bio':
            self.convert_tag = iob2
        elif encoding_type == 'bioes':
            def func(words):
                return iob2bioes(iob2(words))
            # self.convert_tag = lambda words: iob2bioes(iob2(words))
            self.convert_tag = func
        else:
            raise ValueError("encoding_type only supports `bio` and `bioes`.")

        self.bigrams = bigrams
        self.trigrams = trigrams
        self.num_proc = num_proc

    def process(self, data_bundle: DataBundle) -> DataBundle:
        r"""
        支持的DataSet的field为

        .. csv-table::
           :header: "raw_chars", "target"

           "[相, 比, 之, 下,...]", "[O, O, O, O, ...]"
           "[青, 岛, 海, 牛, 队, 和, ...]", "[B-ORG, I-ORG, I-ORG, ...]"
           "[...]", "[...]"

        raw_chars列为List[str], 是未转换的原始数据; chars列为List[int]，是转换为index的输入数据; target列是List[int]，
        是转换为index的target。返回的DataSet中被设置为input有chars, target, seq_len; 设置为target有target。

        :param ~fastNLP.DataBundle data_bundle: 传入的DataBundle中的DataSet必须包含raw_words和ner两个field，且两个field的内容均为List[str]。在传入DataBundle基础上原位修改。
        :return: 处理后的 ``data_bundle``
        """
        # 转换tag
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.convert_tag, field_name='target', new_field_name='target', num_proc=self.num_proc)
        
        _add_chars_field(data_bundle, lower=False)

        input_field_names = ['chars']

        def bigrams(chars):
            return [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])]

        def trigrams(chars):
            return [c1 + c2 + c3 for c1, c2, c3 in
                    zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)]

        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(bigrams, field_name='chars', new_field_name='bigrams', num_proc=self.num_proc)
            input_field_names.append('bigrams')
        if self.trigrams:
            for name, dataset in data_bundle.datasets.items():
                dataset.apply_field(trigrams, field_name='chars', new_field_name='trigrams', num_proc=self.num_proc)
            input_field_names.append('trigrams')

        # index
        _indexize(data_bundle, input_field_names, 'target')
        
        for name, dataset in data_bundle.iter_datasets():
            dataset.add_seq_len('chars')

        return data_bundle


class MsraNERPipe(_CNNERPipe):
    r"""
    处理 **MSRA-NER** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table::
       :header: "raw_chars", "target", "chars", "seq_len"

       "[相, 比, 之, 下,...]", "[0, 0, 0, 0, ...]", "[2, 3, 4, 5, ...]", 11
       "[青, 岛, 海, 牛, 队, 和, ...]", "[1, 2, 3, ...]", "[10, 21, ....]", 21
       "[...]", "[...]", "[...]", .

    ``raw_chars`` 列为 :class:`List` [ :class:`str` ], 是未转换的原始数据； ``chars`` 列为 :class:`List` [ :class:`int` ]，
    是转换为 index 的输入数据； ``target`` 列是 :class:`List` [ :class:`int` ] ，是转换为 index 的 target。返回的 :class:`~fastNLP.core.DataSet` 
    中被设置为 input 有 ``chars`` , ``target``, ``seq_len``；target 有 ``target`` 。

    :param encoding_type: ``target`` 列使用什么类型的 encoding 方式，支持 ``['bioes', 'bio']`` 两种。
    :param bigrams: 是否增加一列 ``bigrams`` 。 ``bigrams`` 会对原文进行如下转化： ``['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]`` 。如果
        设置为 ``True`` ，返回的 :class:`~fastNLP.core.DataSet` 将有一列名为 ``bigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('bigrams')`` 获取。
    :param trigrams: 是否增加一列 ``trigrams`` 。 ``trigrams`` 会对原文进行如下转化 ``['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]`` 。
        如果设置为 ``True`` ，返回的 :class:`~fastNLP.core.DataSet` 将有一列名为 ``trigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('trigrams')`` 获取。
    :param num_proc: 处理数据时使用的进程数目。
    """
    
    def process_from_file(self, paths=None) -> DataBundle:
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load()`

        :param paths:
        :return:
        """
        data_bundle = MsraNERLoader().load(paths)
        return self.process(data_bundle)


class PeopleDailyPipe(_CNNERPipe):
    r"""
    处理 **People's Daily NER** 的 **ner** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table::
       :header: "raw_chars", "target", "chars", "seq_len"

       "[相, 比, 之, 下,...]", "[0, 0, 0, 0, ...]", "[2, 3, 4, 5, ...]", 11
       "[青, 岛, 海, 牛, 队, 和, ...]", "[1, 2, 3, ...]", "[10, 21, ....]", 21
       "[...]", "[...]", "[...]", .

    ``raw_chars`` 列为 :class:`List` [ :class:`str` ], 是未转换的原始数据； ``chars`` 列为 :class:`List` [ :class:`int` ]，
    是转换为 index 的输入数据； ``target`` 列是 :class:`List` [ :class:`int` ] ，是转换为 index 的 target。返回的 :class:`~fastNLP.core.DataSet` 
    中被设置为 input 有 ``chars`` , ``target``, ``seq_len``；target 有 ``target`` 。

    :param encoding_type: ``target`` 列使用什么类型的 encoding 方式，支持 ``['bioes', 'bio']`` 两种。
    :param bigrams: 是否增加一列 ``bigrams`` 。 ``bigrams`` 会对原文进行如下转化： ``['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]`` 。如果
        设置为 ``True`` ，返回的 :class:`~fastNLP.core.DataSet` 将有一列名为 ``bigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('bigrams')`` 获取。
    :param trigrams: 是否增加一列 ``trigrams`` 。 ``trigrams`` 会对原文进行如下转化 ``['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]`` 。
        如果设置为 ``True`` ，返回的 :class:`~fastNLP.core.DataSet` 将有一列名为 ``trigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('trigrams')`` 获取。
    :param num_proc: 处理数据时使用的进程数目。
    """
    
    def process_from_file(self, paths=None) -> DataBundle:
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load()`

        :param paths:
        :return:
        """
        data_bundle = PeopleDailyNERLoader().load(paths)
        return self.process(data_bundle)


class WeiboNERPipe(_CNNERPipe):
    r"""
    处理 **Weibo** 的 **BER** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table::
       :header: "raw_chars", "chars", "target", "seq_len"

       "['老', '百', '姓']", "[4, 3, 3]", "[38, 39, 40]", 3
       "['心']", "[0]", "[41]", 1
       "[...]", "[...]", "[...]", .

    ``raw_chars`` 列为 :class:`List` [ :class:`str` ], 是未转换的原始数据； ``chars`` 列为 :class:`List` [ :class:`int` ]，
    是转换为 index 的输入数据； ``target`` 列是 :class:`List` [ :class:`int` ] ，是转换为 index 的 target。返回的 :class:`~fastNLP.core.DataSet` 
    中被设置为 input 有 ``chars`` , ``target``, ``seq_len``；target 有 ``target`` 。

    :param encoding_type: ``target`` 列使用什么类型的 encoding 方式，支持 ``['bioes', 'bio']`` 两种。
    :param bigrams: 是否增加一列 ``bigrams`` 。 ``bigrams`` 会对原文进行如下转化： ``['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]`` 。如果
        设置为 ``True`` ，返回的 :class:`~fastNLP.core.DataSet` 将有一列名为 ``bigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('bigrams')`` 获取。
    :param trigrams: 是否增加一列 ``trigrams`` 。 ``trigrams`` 会对原文进行如下转化 ``['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]`` 。
        如果设置为 ``True`` ，返回的 :class:`~fastNLP.core.DataSet` 将有一列名为 ``trigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('trigrams')`` 获取。
    :param num_proc: 处理数据时使用的进程数目。
    """
    
    def process_from_file(self, paths=None) -> DataBundle:
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load()`

        :param paths:
        :return:
        """
        data_bundle = WeiboNERLoader().load(paths)
        return self.process(data_bundle)
