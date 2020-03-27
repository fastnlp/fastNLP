r"""undocumented"""

__all__ = [
    "CLSBasePipe",
    "AGsNewsPipe",
    "DBPediaPipe",
    "YelpFullPipe",
    "YelpPolarityPipe",
    "SSTPipe",
    "SST2Pipe",
    'IMDBPipe',
    "ChnSentiCorpPipe",
    "THUCNewsPipe",
    "WeiboSenti100kPipe"
]

import re
import warnings

from nltk import Tree

from .pipe import Pipe
from .utils import get_tokenizer, _indexize, _add_words_field, _add_chars_field, _granularize
from ..data_bundle import DataBundle
from ..loader.classification import ChnSentiCorpLoader, THUCNewsLoader, WeiboSenti100kLoader
from ..loader.classification import IMDBLoader, YelpFullLoader, SSTLoader, SST2Loader, YelpPolarityLoader, \
    AGsNewsLoader, DBPediaLoader
from ...core._logger import logger
from ...core.const import Const
from ...core.dataset import DataSet
from ...core.instance import Instance


class CLSBasePipe(Pipe):

    def __init__(self, lower: bool=False, tokenizer: str='spacy', lang='en'):
        super().__init__()
        self.lower = lower
        self.tokenizer = get_tokenizer(tokenizer, lang=lang)

    def _tokenize(self, data_bundle, field_name=Const.INPUT, new_field_name=None):
        r"""
        将DataBundle中的数据进行tokenize

        :param DataBundle data_bundle:
        :param str field_name:
        :param str new_field_name:
        :return: 传入的DataBundle对象
        """
        new_field_name = new_field_name or field_name
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.tokenizer, field_name=field_name, new_field_name=new_field_name)

        return data_bundle

    def process(self, data_bundle: DataBundle):
        r"""
        传入的DataSet应该具备如下的结构

        .. csv-table::
            :header: "raw_words", "target"

            "I got 'new' tires from them and... ", "1"
            "Don't waste your time.  We had two...", "1"
            "...", "..."

        :param data_bundle:
        :return:
        """
        # 复制一列words
        data_bundle = _add_words_field(data_bundle, lower=self.lower)
        # 进行tokenize
        data_bundle = self._tokenize(data_bundle=data_bundle, field_name=Const.INPUT)
        # 建立词表并index
        data_bundle = _indexize(data_bundle=data_bundle)

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)

        data_bundle.set_input(Const.INPUT, Const.INPUT_LEN)
        data_bundle.set_target(Const.TARGET)

        return data_bundle

    def process_from_file(self, paths) -> DataBundle:
        r"""
        传入文件路径，生成处理好的DataBundle对象。paths支持的路径形式可以参考 ：:meth:`fastNLP.io.Loader.load()`

        :param paths:
        :return: DataBundle
        """
        raise NotImplementedError


class YelpFullPipe(CLSBasePipe):
    r"""
    处理YelpFull的数据, 处理之后DataSet中的内容如下

    .. csv-table:: 下面是使用YelpFullPipe处理后的DataSet所具备的field
        :header: "raw_words", "target", "words",  "seq_len"

        "I got 'new' tires from them and within...", 0 ,"[7, 110, 22, 107, 22, 499, 59, 140, 3,...]", 160
        " Don't waste your time.  We had two dif... ", 0, "[277, 17, 278, 38, 30, 112, 24, 85, 27...", 40
        "...", ., "[...]", .

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_words | target | words | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   | False  |  True |   True  |
        |  is_target  |   False   |  True  | False |  False  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    """
    
    def __init__(self, lower: bool = False, granularity=5, tokenizer: str = 'spacy'):
        r"""
        
        :param bool lower: 是否对输入进行小写化。
        :param int granularity: 支持2, 3, 5。若为2, 则认为是2分类问题，将1、2归为1类，4、5归为一类，丢掉2；若为3, 则有3分类问题，将
            1、2归为1类，3归为1类，4、5归为1类；若为5, 则有5分类问题。
        :param str tokenizer: 使用哪种tokenize方式将数据切成单词。支持'spacy'和'raw'。raw使用空格作为切分。
        """
        super().__init__(lower=lower, tokenizer=tokenizer, lang='en')
        assert granularity in (2, 3, 5), "granularity can only be 2,3,5."
        self.granularity = granularity
        
        if granularity == 2:
            self.tag_map = {"1": "negative", "2": "negative", "4": "positive", "5": "positive"}
        elif granularity == 3:
            self.tag_map = {"1": "negative", "2": "negative", "3": "medium", "4": "positive", "5": "positive"}
        else:
            self.tag_map = None
    
    def process(self, data_bundle):
        r"""
        传入的DataSet应该具备如下的结构

        .. csv-table::
           :header: "raw_words", "target"

           "I got 'new' tires from them and... ", "1"
           "Don't waste your time.  We had two...", "1"
           "...", "..."

        :param data_bundle:
        :return:
        """
        if self.tag_map is not None:
            data_bundle = _granularize(data_bundle, self.tag_map)

        data_bundle = super().process(data_bundle)
        
        return data_bundle
    
    def process_from_file(self, paths=None):
        r"""

        :param paths:
        :return: DataBundle
        """
        data_bundle = YelpFullLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class YelpPolarityPipe(CLSBasePipe):
    r"""
    处理YelpPolarity的数据, 处理之后DataSet中的内容如下

    .. csv-table:: 下面是使用YelpFullPipe处理后的DataSet所具备的field
        :header: "raw_words", "target", "words", "seq_len"

        "I got 'new' tires from them and within...", 0 ,"[7, 110, 22, 107, 22, 499, 59, 140, 3,...]", 160
        " Don't waste your time.  We had two dif... ", 0, "[277, 17, 278, 38, 30, 112, 24, 85, 27...", 40
        "...", ., "[...]", .

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_words | target | words | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   | False  |  True |   True  |
        |  is_target  |   False   |  True  | False |  False  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    """
    
    def __init__(self, lower: bool = False, tokenizer: str = 'spacy'):
        r"""
        
        :param bool lower: 是否对输入进行小写化。
        :param str tokenizer: 使用哪种tokenize方式将数据切成单词。支持'spacy'和'raw'。raw使用空格作为切分。
        """
        super().__init__(lower=lower, tokenizer=tokenizer, lang='en')
    
    def process_from_file(self, paths=None):
        r"""

        :param str paths:
        :return: DataBundle
        """
        data_bundle = YelpPolarityLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class AGsNewsPipe(CLSBasePipe):
    r"""
    处理AG's News的数据, 处理之后DataSet中的内容如下

    .. csv-table:: 下面是使用AGsNewsPipe处理后的DataSet所具备的field
        :header: "raw_words", "target", "words", "seq_len"

        "I got 'new' tires from them and within...", 0 ,"[7, 110, 22, 107, 22, 499, 59, 140, 3,...]", 160
        " Don't waste your time.  We had two dif... ", 0, "[277, 17, 278, 38, 30, 112, 24, 85, 27...", 40
        "...", ., "[...]", .

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_words | target | words | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   | False  |  True |   True  |
        |  is_target  |   False   |  True  | False |  False  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    """

    def __init__(self, lower: bool = False, tokenizer: str = 'spacy'):
        r"""

        :param bool lower: 是否对输入进行小写化。
        :param str tokenizer: 使用哪种tokenize方式将数据切成单词。支持'spacy'和'raw'。raw使用空格作为切分。
        """
        super().__init__(lower=lower, tokenizer=tokenizer, lang='en')

    def process_from_file(self, paths=None):
        r"""
        :param str paths:
        :return: DataBundle
        """
        data_bundle = AGsNewsLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class DBPediaPipe(CLSBasePipe):
    r"""
    处理DBPedia的数据, 处理之后DataSet中的内容如下

    .. csv-table:: 下面是使用DBPediaPipe处理后的DataSet所具备的field
        :header: "raw_words", "target", "words", "seq_len"

        "I got 'new' tires from them and within...", 0 ,"[7, 110, 22, 107, 22, 499, 59, 140, 3,...]", 160
        " Don't waste your time.  We had two dif... ", 0, "[277, 17, 278, 38, 30, 112, 24, 85, 27...", 40
        "...", ., "[...]", .

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_words | target | words | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   | False  |  True |   True  |
        |  is_target  |   False   |  True  | False |  False  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    """

    def __init__(self, lower: bool = False, tokenizer: str = 'spacy'):
        r"""

        :param bool lower: 是否对输入进行小写化。
        :param str tokenizer: 使用哪种tokenize方式将数据切成单词。支持'spacy'和'raw'。raw使用空格作为切分。
        """
        super().__init__(lower=lower, tokenizer=tokenizer, lang='en')

    def process_from_file(self, paths=None):
        r"""
        :param str paths:
        :return: DataBundle
        """
        data_bundle = DBPediaLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class SSTPipe(CLSBasePipe):
    r"""
    经过该Pipe之后，DataSet中具备的field如下所示

    .. csv-table:: 下面是使用SSTPipe处理后的DataSet所具备的field
        :header: "raw_words", "words", "target", "seq_len"

        "It 's a lovely film with lovely perfor...", 1, "[187, 6, 5, 132, 120, 70, 132, 188, 25...", 13
        "No one goes unindicted here , which is...", 0, "[191, 126, 192, 193, 194, 4, 195, 17, ...", 13
        "...", ., "[...]", .

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_words | target | words | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   | False  |  True |   True  |
        |  is_target  |   False   |  True  | False |  False  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    """
    
    def __init__(self, subtree=False, train_subtree=True, lower=False, granularity=5, tokenizer='spacy'):
        r"""
        
        :param bool subtree: 是否将train, test, dev数据展开为子树，扩充数据量。 Default: ``False``
        :param bool train_subtree: 是否将train集通过子树扩展数据。
        :param bool lower: 是否对输入进行小写化。
        :param int granularity: 支持2, 3, 5。若为2, 则认为是2分类问题，将0、1归为1类，3、4归为一类，丢掉2；若为3, 则有3分类问题，将
            0、1归为1类，2归为1类，3、4归为1类；若为5, 则有5分类问题。
        :param str tokenizer: 使用哪种tokenize方式将数据切成单词。支持'spacy'和'raw'。raw使用空格作为切分。
        """
        super().__init__(tokenizer=tokenizer, lang='en')
        self.subtree = subtree
        self.train_tree = train_subtree
        self.lower = lower
        assert granularity in (2, 3, 5), "granularity can only be 2,3,5."
        self.granularity = granularity
        
        if granularity == 2:
            self.tag_map = {"0": "negative", "1": "negative", "3": "positive", "4": "positive"}
        elif granularity == 3:
            self.tag_map = {"0": "negative", "1": "negative", "2": "medium", "3": "positive", "4": "positive"}
        else:
            self.tag_map = None
    
    def process(self, data_bundle: DataBundle):
        r"""
        对DataBundle中的数据进行预处理。输入的DataSet应该至少拥有raw_words这一列，且内容类似与

        .. csv-table:: 下面是使用SSTLoader读取的DataSet所具备的field
            :header: "raw_words"

            "(2 (3 (3 Effective) (2 but)) (1 (1 too-tepid)..."
            "(3 (3 (2 If) (3 (2 you) (3 (2 sometimes) ..."
            "..."

        :param ~fastNLP.io.DataBundle data_bundle: 需要处理的DataBundle对象
        :return:
        """
        #  先取出subtree
        for name in list(data_bundle.datasets.keys()):
            dataset = data_bundle.get_dataset(name)
            ds = DataSet()
            use_subtree = self.subtree or (name == 'train' and self.train_tree)
            for ins in dataset:
                raw_words = ins[Const.RAW_WORD]
                tree = Tree.fromstring(raw_words)
                if use_subtree:
                    for t in tree.subtrees():
                        raw_words = " ".join(t.leaves())
                        instance = Instance(raw_words=raw_words, target=t.label())
                        ds.append(instance)
                else:
                    instance = Instance(raw_words=' '.join(tree.leaves()), target=tree.label())
                    ds.append(instance)
            data_bundle.set_dataset(ds, name)

        # 根据granularity设置tag
        data_bundle = _granularize(data_bundle, tag_map=self.tag_map)
        
        data_bundle = super().process(data_bundle)
        
        return data_bundle
    
    def process_from_file(self, paths=None):
        data_bundle = SSTLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class SST2Pipe(CLSBasePipe):
    r"""
    加载SST2的数据, 处理完成之后DataSet将拥有以下的field

    .. csv-table::
       :header: "raw_words", "target", "words", "seq_len"

       "it 's a charming and often affecting j... ", 1, "[19, 9, 6, 111, 5, 112, 113, 114, 3]", 9
       "unflinchingly bleak and desperate", 0, "[115, 116, 5, 117]", 4
       "...", "...", ., .

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_words | target | words | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   | False  |  True |   True  |
        |  is_target  |   False   |  True  | False |  False  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    """
    
    def __init__(self, lower=False, tokenizer='spacy'):
        r"""
        
        :param bool lower: 是否对输入进行小写化。
        :param str tokenizer: 使用哪种tokenize方式将数据切成单词。支持'spacy'和'raw'。raw使用空格作为切分。
        """
        super().__init__(lower=lower, tokenizer=tokenizer, lang='en')
    
    def process_from_file(self, paths=None):
        r"""

        :param str paths: 如果为None，则自动下载并缓存到fastNLP的缓存地址。
        :return: DataBundle
        """
        data_bundle = SST2Loader().load(paths)
        return self.process(data_bundle)


class IMDBPipe(CLSBasePipe):
    r"""
    经过本Pipe处理后DataSet将如下

    .. csv-table:: 输出DataSet的field
       :header: "raw_words", "target", "words", "seq_len"

       "Bromwell High is a cartoon ... ", 0, "[3, 5, 6, 9, ...]", 20
       "Story of a man who has ...", 1, "[20, 43, 9, 10, ...]", 31
       "...", ., "[...]", .

    其中raw_words为str类型，是原文; words是转换为index的输入; target是转换为index的目标值;
    words列被设置为input; target列被设置为target。

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_words | target | words | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   | False  |  True |   True  |
        |  is_target  |   False   |  True  | False |  False  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    """
    
    def __init__(self, lower: bool = False, tokenizer: str = 'spacy'):
        r"""
        
        :param bool lower: 是否将words列的数据小写。
        :param str tokenizer: 使用什么tokenizer来将句子切分为words. 支持spacy, raw两种。raw即使用空格拆分。
        """
        super().__init__(tokenizer=tokenizer, lang='en')
        self.lower = lower
    
    def process(self, data_bundle: DataBundle):
        r"""
        期待的DataBunlde中输入的DataSet应该类似于如下，有两个field，raw_words和target，且均为str类型

        .. csv-table:: 输入DataSet的field
           :header: "raw_words", "target"

           "Bromwell High is a cartoon ... ", "pos"
           "Story of a man who has ...", "neg"
           "...", "..."

        :param DataBunlde data_bundle: 传入的DataBundle中的DataSet必须包含raw_words和target两个field，且raw_words列应该为str,
            target列应该为str。
        :return: DataBundle
        """
        
        # 替换<br />
        def replace_br(raw_words):
            raw_words = raw_words.replace("<br />", ' ')
            return raw_words
        
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(replace_br, field_name=Const.RAW_WORD, new_field_name=Const.RAW_WORD)
        
        data_bundle = super().process(data_bundle)
        
        return data_bundle
    
    def process_from_file(self, paths=None):
        r"""

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.Loader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = IMDBLoader().load(paths)
        data_bundle = self.process(data_bundle)
        
        return data_bundle


class ChnSentiCorpPipe(Pipe):
    r"""
    处理之后的DataSet有以下的结构

    .. csv-table::
        :header: "raw_chars", "target", "chars", "seq_len"

        "這間酒店環境和服務態度亦算不錯,但房間空間太小~~", 1, "[2, 3, 4, 5, ...]", 31
        "<荐书> 推荐所有喜欢<红楼>...", 1, "[10, 21, ....]", 25
        "..."

    其中chars, seq_len是input，target是target
    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_chars | target | chars | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   |  True  |  True |   True  |
        |  is_target  |   False   |  True  | False |  False  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    """
    def __init__(self, bigrams=False, trigrams=False):
        r"""
        
        :param bool bigrams: 是否增加一列bigrams. bigrams的构成是['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]。如果
            设置为True，返回的DataSet将有一列名为bigrams, 且已经转换为了index并设置为input，对应的vocab可以通过
            data_bundle.get_vocab('bigrams')获取.
        :param bool trigrams: 是否增加一列trigrams. trigrams的构成是 ['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]
            。如果设置为True，返回的DataSet将有一列名为trigrams, 且已经转换为了index并设置为input，对应的vocab可以通过
            data_bundle.get_vocab('trigrams')获取.
        """
        super().__init__()

        self.bigrams = bigrams
        self.trigrams = trigrams

    def _tokenize(self, data_bundle):
        r"""
        将DataSet中的"复旦大学"拆分为["复", "旦", "大", "学"]. 未来可以通过扩展这个函数实现分词。

        :param data_bundle:
        :return:
        """
        data_bundle.apply_field(list, field_name=Const.CHAR_INPUT, new_field_name=Const.CHAR_INPUT)
        return data_bundle

    def process(self, data_bundle:DataBundle):
        r"""
        可以处理的DataSet应该具备以下的field

        .. csv-table::
            :header: "raw_chars", "target"

            "這間酒店環境和服務態度亦算不錯,但房間空間太小~~", "1"
            "<荐书> 推荐所有喜欢<红楼>...", "1"
            "..."

        :param data_bundle:
        :return:
        """
        _add_chars_field(data_bundle, lower=False)

        data_bundle = self._tokenize(data_bundle)

        input_field_names = [Const.CHAR_INPUT]
        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT, new_field_name='bigrams')
            input_field_names.append('bigrams')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT, new_field_name='trigrams')
            input_field_names.append('trigrams')

        # index
        _indexize(data_bundle, input_field_names, Const.TARGET)

        input_fields = [Const.TARGET, Const.INPUT_LEN] + input_field_names
        target_fields = [Const.TARGET]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.CHAR_INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths=None):
        r"""

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.Loader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = ChnSentiCorpLoader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class THUCNewsPipe(CLSBasePipe):
    r"""
    处理之后的DataSet有以下的结构

    .. csv-table::
        :header: "raw_chars", "target", "chars", "seq_len"

        "马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道...", 0, "[409, 1197, 2146, 213, ...]", 746
        "..."

    其中chars, seq_len是input，target是target
    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_chars | target | chars | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   |  True  |  True |   True  |
        |  is_target  |   False   |  True  | False |  False  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

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
        # return [w for w in sent]

    def _raw_split(self, sent):
        return sent.split()

    def _tokenize(self, data_bundle, field_name=Const.INPUT, new_field_name=None):
        new_field_name = new_field_name or field_name
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self._chracter_split, field_name=field_name, new_field_name=new_field_name)
        return data_bundle

    def process(self, data_bundle: DataBundle):
        r"""
        可处理的DataSet应具备如下的field

        .. csv-table::
            :header: "raw_words", "target"
            
            "马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 ... ", "体育"
            "...", "..."

        :param data_bundle:
        :return:
        """
        # 根据granularity设置tag
        tag_map = {'体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4, '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9}
        data_bundle = _granularize(data_bundle=data_bundle, tag_map=tag_map)

        # clean,lower

        # CWS(tokenize)
        data_bundle = self._tokenize(data_bundle=data_bundle, field_name='raw_chars', new_field_name='chars')

        input_field_names = [Const.CHAR_INPUT]

        # n-grams
        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT, new_field_name='bigrams')
            input_field_names.append('bigrams')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT, new_field_name='trigrams')
            input_field_names.append('trigrams')

        # index
        data_bundle = _indexize(data_bundle=data_bundle, input_field_names=Const.CHAR_INPUT)

        # add length
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(field_name=Const.CHAR_INPUT, new_field_name=Const.INPUT_LEN)

        input_fields = [Const.TARGET, Const.INPUT_LEN] + input_field_names
        target_fields = [Const.TARGET]

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths=None):
        r"""
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.Loader` 的load函数。
        :return: DataBundle
        """
        data_loader = THUCNewsLoader()  # 此处需要实例化一个data_loader，否则传入load()的参数为None
        data_bundle = data_loader.load(paths)
        data_bundle = self.process(data_bundle)
        return data_bundle


class WeiboSenti100kPipe(CLSBasePipe):
    r"""
    处理之后的DataSet有以下的结构

    .. csv-table::
        :header: "raw_chars", "target", "chars", "seq_len"

        "六一出生的？好讽刺…… //@祭春姬:他爸爸是外星人吧 //@面孔小高:现在的孩子都怎么了 [怒][怒][怒]", 0, "[0, 690, 18, ...]", 56
        "..."

    其中chars, seq_len是input，target是target
    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_chars | target | chars | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   |  True  |  True |   True  |
        |  is_target  |   False   |  True  | False |  False  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

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

    def _tokenize(self, data_bundle, field_name=Const.INPUT, new_field_name=None):
        new_field_name = new_field_name or field_name
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self._chracter_split, field_name=field_name, new_field_name=new_field_name)
        return data_bundle

    def process(self, data_bundle: DataBundle):
        r"""
        可处理的DataSet应具备以下的field

        .. csv-table::
            :header: "raw_chars", "target"
            
            "六一出生的？好讽刺…… //@祭春姬:他爸爸是外星人吧 //@面孔小高:现在的孩子都怎么了 [怒][怒][怒]", "0"
            "...", "..."

        :param data_bundle:
        :return:
        """
        # clean,lower

        # CWS(tokenize)
        data_bundle = self._tokenize(data_bundle=data_bundle, field_name='raw_chars', new_field_name='chars')

        input_field_names = [Const.CHAR_INPUT]

        # n-grams
        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT, new_field_name='bigrams')
            input_field_names.append('bigrams')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT, new_field_name='trigrams')
            input_field_names.append('trigrams')

        # index
        data_bundle = _indexize(data_bundle=data_bundle, input_field_names='chars')

        # add length
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(field_name=Const.CHAR_INPUT, new_field_name=Const.INPUT_LEN)

        input_fields = [Const.TARGET, Const.INPUT_LEN] + input_field_names
        target_fields = [Const.TARGET]

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths=None):
        r"""
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.Loader` 的load函数。
        :return: DataBundle
        """
        data_loader = WeiboSenti100kLoader()  # 此处需要实例化一个data_loader，否则传入load()的参数为None
        data_bundle = data_loader.load(paths)
        data_bundle = self.process(data_bundle)
        return data_bundle

