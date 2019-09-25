"""undocumented"""

__all__ = [
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
from .utils import get_tokenizer, _indexize, _add_words_field, _drop_empty_instance, _add_chars_field
from ..data_bundle import DataBundle
from ..loader.classification import ChnSentiCorpLoader, THUCNewsLoader, WeiboSenti100kLoader
from ..loader.classification import IMDBLoader, YelpFullLoader, SSTLoader, SST2Loader, YelpPolarityLoader
from ...core._logger import logger
from ...core.const import Const
from ...core.dataset import DataSet
from ...core.instance import Instance
from ...core.vocabulary import Vocabulary

nonalpnum = re.compile('[^0-9a-zA-Z?!\']+')


class _CLSPipe(Pipe):
    """
    分类问题的基类，负责对classification的数据进行tokenize操作。默认是对raw_words列操作，然后生成words列

    """
    
    def __init__(self, tokenizer: str = 'spacy', lang='en'):
        
        self.tokenizer = get_tokenizer(tokenizer, lang=lang)
    
    def _tokenize(self, data_bundle, field_name=Const.INPUT, new_field_name=None):
        """
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


def _clean_str(words):
    """
    heavily borrowed from github
    https://github.com/LukeZhuang/Hierarchical-Attention-Network/blob/master/yelp-preprocess.ipynb
    :param sentence:  is a str
    :return:
    """
    words_collection = []
    for word in words:
        if word in ['-lrb-', '-rrb-', '<sssss>', '-r', '-l', 'b-']:
            continue
        tt = nonalpnum.split(word)
        t = ''.join(tt)
        if t != '':
            words_collection.append(t)
    
    return words_collection


class YelpFullPipe(_CLSPipe):
    """
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
        """
        
        :param bool lower: 是否对输入进行小写化。
        :param int granularity: 支持2, 3, 5。若为2, 则认为是2分类问题，将1、2归为1类，4、5归为一类，丢掉2；若为3, 则有3分类问题，将
            1、2归为1类，3归为1类，4、5归为1类；若为5, 则有5分类问题。
        :param str tokenizer: 使用哪种tokenize方式将数据切成单词。支持'spacy'和'raw'。raw使用空格作为切分。
        """
        super().__init__(tokenizer=tokenizer, lang='en')
        self.lower = lower
        assert granularity in (2, 3, 5), "granularity can only be 2,3,5."
        self.granularity = granularity
        
        if granularity == 2:
            self.tag_map = {"1": 0, "2": 0, "4": 1, "5": 1}
        elif granularity == 3:
            self.tag_map = {"1": 0, "2": 0, "3": 1, "4": 2, "5": 2}
        else:
            self.tag_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
    
    def _tokenize(self, data_bundle, field_name=Const.INPUT, new_field_name=None):
        """
        将DataBundle中的数据进行tokenize

        :param DataBundle data_bundle:
        :param str field_name:
        :param str new_field_name:
        :return: 传入的DataBundle对象
        """
        new_field_name = new_field_name or field_name
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.tokenizer, field_name=field_name, new_field_name=new_field_name)
            dataset.apply_field(_clean_str, field_name=field_name, new_field_name=new_field_name)
        return data_bundle
    
    def process(self, data_bundle):
        """
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
        
        # 根据granularity设置tag
        data_bundle = self._granularize(data_bundle, tag_map=self.tag_map)
        
        # 删除空行
        data_bundle = _drop_empty_instance(data_bundle, field_name=Const.INPUT)
        
        # index
        data_bundle = _indexize(data_bundle=data_bundle)
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
        
        data_bundle.set_input(Const.INPUT, Const.INPUT_LEN)
        data_bundle.set_target(Const.TARGET)
        
        return data_bundle
    
    def process_from_file(self, paths=None):
        """

        :param paths:
        :return: DataBundle
        """
        data_bundle = YelpFullLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class YelpPolarityPipe(_CLSPipe):
    """
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
        """
        
        :param bool lower: 是否对输入进行小写化。
        :param str tokenizer: 使用哪种tokenize方式将数据切成单词。支持'spacy'和'raw'。raw使用空格作为切分。
        """
        super().__init__(tokenizer=tokenizer, lang='en')
        self.lower = lower
    
    def process(self, data_bundle):
        """
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
        # index
        data_bundle = _indexize(data_bundle=data_bundle)
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
        
        data_bundle.set_input(Const.INPUT, Const.INPUT_LEN)
        data_bundle.set_target(Const.TARGET)
        
        return data_bundle
    
    def process_from_file(self, paths=None):
        """

        :param str paths:
        :return: DataBundle
        """
        data_bundle = YelpPolarityLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class SSTPipe(_CLSPipe):
    """
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
        """
        
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
            self.tag_map = {"0": 0, "1": 0, "3": 1, "4": 1}
        elif granularity == 3:
            self.tag_map = {"0": 0, "1": 0, "2": 1, "3": 2, "4": 2}
        else:
            self.tag_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
    
    def process(self, data_bundle: DataBundle):
        """
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
                raw_words = ins['raw_words']
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
        
        _add_words_field(data_bundle, lower=self.lower)
        
        # 进行tokenize
        data_bundle = self._tokenize(data_bundle=data_bundle, field_name=Const.INPUT)
        
        # 根据granularity设置tag
        data_bundle = self._granularize(data_bundle, tag_map=self.tag_map)
        
        # index
        data_bundle = _indexize(data_bundle=data_bundle)
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
        
        data_bundle.set_input(Const.INPUT, Const.INPUT_LEN)
        data_bundle.set_target(Const.TARGET)
        
        return data_bundle
    
    def process_from_file(self, paths=None):
        data_bundle = SSTLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class SST2Pipe(_CLSPipe):
    """
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
        """
        
        :param bool lower: 是否对输入进行小写化。
        :param str tokenizer: 使用哪种tokenize方式将数据切成单词。支持'spacy'和'raw'。raw使用空格作为切分。
        """
        super().__init__(tokenizer=tokenizer, lang='en')
        self.lower = lower
    
    def process(self, data_bundle: DataBundle):
        """
        可以处理的DataSet应该具备如下的结构

        .. csv-table::
            :header: "raw_words", "target"

            "it 's a charming and often affecting...", "1"
            "unflinchingly bleak and...", "0"
            "..."

        :param data_bundle:
        :return:
        """
        _add_words_field(data_bundle, self.lower)
        
        data_bundle = self._tokenize(data_bundle=data_bundle)
        
        src_vocab = Vocabulary()
        src_vocab.from_dataset(data_bundle.datasets['train'], field_name=Const.INPUT,
                               no_create_entry_dataset=[dataset for name, dataset in data_bundle.datasets.items() if
                                                        name != 'train'])
        src_vocab.index_dataset(*data_bundle.datasets.values(), field_name=Const.INPUT)
        
        tgt_vocab = Vocabulary(unknown=None, padding=None)
        tgt_vocab.from_dataset(*[ds for name, ds in data_bundle.iter_datasets() if 'train' in name],
                               field_name=Const.TARGET,
                               no_create_entry_dataset=[ds for name, ds in data_bundle.iter_datasets()
                                                        if ('train' not in name) and (ds.has_field(Const.TARGET))]
                               )
        if len(tgt_vocab._no_create_word) > 0:
            warn_msg = f"There are {len(tgt_vocab._no_create_word)} target labels" \
                       f" in {[name for name in data_bundle.datasets.keys() if 'train' not in name]} " \
                       f"data set but not in train data set!."
            warnings.warn(warn_msg)
            logger.warning(warn_msg)
        datasets = []
        for name, dataset in data_bundle.datasets.items():
            if dataset.has_field(Const.TARGET):
                datasets.append(dataset)
        tgt_vocab.index_dataset(*datasets, field_name=Const.TARGET)
        
        data_bundle.set_vocab(src_vocab, Const.INPUT)
        data_bundle.set_vocab(tgt_vocab, Const.TARGET)
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
        
        data_bundle.set_input(Const.INPUT, Const.INPUT_LEN)
        data_bundle.set_target(Const.TARGET)
        
        return data_bundle
    
    def process_from_file(self, paths=None):
        """

        :param str paths: 如果为None，则自动下载并缓存到fastNLP的缓存地址。
        :return: DataBundle
        """
        data_bundle = SST2Loader().load(paths)
        return self.process(data_bundle)


class IMDBPipe(_CLSPipe):
    """
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
        """
        
        :param bool lower: 是否将words列的数据小写。
        :param str tokenizer: 使用什么tokenizer来将句子切分为words. 支持spacy, raw两种。raw即使用空格拆分。
        """
        super().__init__(tokenizer=tokenizer, lang='en')
        self.lower = lower
    
    def process(self, data_bundle: DataBundle):
        """
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
        
        _add_words_field(data_bundle, lower=self.lower)
        self._tokenize(data_bundle, field_name=Const.INPUT, new_field_name=Const.INPUT)
        _indexize(data_bundle)
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
            dataset.set_input(Const.INPUT, Const.INPUT_LEN)
            dataset.set_target(Const.TARGET)
        
        return data_bundle
    
    def process_from_file(self, paths=None):
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.Loader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = IMDBLoader().load(paths)
        data_bundle = self.process(data_bundle)
        
        return data_bundle


class ChnSentiCorpPipe(Pipe):
    """
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
        """
        
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
        """
        将DataSet中的"复旦大学"拆分为["复", "旦", "大", "学"]. 未来可以通过扩展这个函数实现分词。

        :param data_bundle:
        :return:
        """
        data_bundle.apply_field(list, field_name=Const.CHAR_INPUT, new_field_name=Const.CHAR_INPUT)
        return data_bundle

    def process(self, data_bundle:DataBundle):
        """
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
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.Loader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = ChnSentiCorpLoader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class THUCNewsPipe(_CLSPipe):
    """
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
        """
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
        data_bundle = self._granularize(data_bundle=data_bundle, tag_map=tag_map)

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
        """
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.Loader` 的load函数。
        :return: DataBundle
        """
        data_loader = THUCNewsLoader()  # 此处需要实例化一个data_loader，否则传入load()的参数为None
        data_bundle = data_loader.load(paths)
        data_bundle = self.process(data_bundle)
        return data_bundle


class WeiboSenti100kPipe(_CLSPipe):
    """
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
        """
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
        """
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.Loader` 的load函数。
        :return: DataBundle
        """
        data_loader = WeiboSenti100kLoader()  # 此处需要实例化一个data_loader，否则传入load()的参数为None
        data_bundle = data_loader.load(paths)
        data_bundle = self.process(data_bundle)
        return data_bundle

