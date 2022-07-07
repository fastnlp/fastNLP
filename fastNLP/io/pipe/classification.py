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
    "WeiboSenti100kPipe",
    "MRPipe", "R8Pipe", "R52Pipe", "OhsumedPipe", "NG20Pipe"
]

import re

try:
    from nltk import Tree
except:
    # only nltk in some versions can run
    pass

from .pipe import Pipe
from .utils import get_tokenizer, _indexize, _add_words_field, _add_chars_field, _granularize
from ..data_bundle import DataBundle
from ..loader.classification import ChnSentiCorpLoader, THUCNewsLoader, WeiboSenti100kLoader
from ..loader.classification import IMDBLoader, YelpFullLoader, SSTLoader, SST2Loader, YelpPolarityLoader, \
    AGsNewsLoader, DBPediaLoader, MRLoader, R52Loader, R8Loader, OhsumedLoader, NG20Loader
# from ...core._logger import log
# from ...core.const import Const
from fastNLP.core.dataset import DataSet, Instance
from fastNLP.core.log import logger


class CLSBasePipe(Pipe):
    """
    处理分类数据集 **Pipe** 的基类。

    :param lower: 是否对输入进行小写化。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw', 'cn-char']`` 。``'raw'`` 表示使用空格作为切分， ``'cn-char'`` 表示
        按字符切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param lang: :mod:`spacy` 使用的语言，当前仅支持 ``'en'`` 。
    :param num_proc: 处理数据时使用的进程数目。
    """
    
    def __init__(self, lower: bool = False, tokenizer: str = 'raw', lang: str='en', num_proc: int=0):
        super().__init__()
        self.lower = lower
        self.tokenizer = get_tokenizer(tokenizer, lang=lang)
        self.num_proc = num_proc
    
    def _tokenize(self, data_bundle, field_name='words', new_field_name=None):
        r"""
        将DataBundle中的数据进行tokenize

        :param DataBundle data_bundle:
        :param str field_name:
        :param str new_field_name:
        :return: 传入的DataBundle对象
        """
        new_field_name = new_field_name or field_name
        for name, dataset in data_bundle.iter_datasets():
            dataset.apply_field(self.tokenizer, field_name=field_name, new_field_name=new_field_name,
                                num_proc=self.num_proc)

        return data_bundle

    def process(self, data_bundle: DataBundle):
        r"""
        ``data_bunlde`` 中的 :class:`~fastNLP.core.DataSet` 应该具备如下的结构：

        .. csv-table::
            :header: "raw_words", "target"

            "I got 'new' tires from them and... ", "1"
            "Don't waste your time.  We had two...", "1"
            "...", "..."

        :param data_bundle:
        :return: 处理后的 ``data_bundle``
        """
        # 复制一列words
        data_bundle = _add_words_field(data_bundle, lower=self.lower)
        # 进行tokenize
        data_bundle = self._tokenize(data_bundle=data_bundle, field_name='words')
        # 建立词表并index
        data_bundle = _indexize(data_bundle=data_bundle)

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len('words')

        return data_bundle

    def process_from_file(self, paths) -> DataBundle:
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        raise NotImplementedError


class YelpFullPipe(CLSBasePipe):
    r"""
    处理 **Yelp Review Full** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table:: 下面是使用 YelpFullPipe 处理后的 DataSet 所具备的 field
        :header: "raw_words", "target", "words",  "seq_len"

        "I got 'new' tires from them and within...", 0 ,"[7, 110, 22, 107, 22, 499, 59, 140, 3,...]", 160
        " Don't waste your time.  We had two dif... ", 0, "[277, 17, 278, 38, 30, 112, 24, 85, 27...", 40
        "...", ., "[...]", .

    :param lower: 是否对输入进行小写化。
    :param granularity: 支持 ``[2, 3, 5]`` 。若为 ``2`` ，则认为是二分类问题，将 **1、2** 归为一类， **4、5** 归为一类，
        丢掉 3；若为 ``3`` ，则认为是三分类问题，将 **1、2** 归为一类， **3** 归为一类， **4、5** 归为一类；若为 ``5`` ，则认为是五分类问题。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw']`` 。``'raw'`` 表示使用空格作为切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param num_proc: 处理数据时使用的进程数目。
    """
    def __init__(self, lower: bool = False, granularity: int=5, tokenizer: str = 'spacy', num_proc: int=0):
        super().__init__(lower=lower, tokenizer=tokenizer, lang='en', num_proc=num_proc)
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
        ``data_bunlde`` 中的 :class:`~fastNLP.core.DataSet` 应该具备如下的结构：

        .. csv-table::
           :header: "raw_words", "target"

           "I got 'new' tires from them and... ", "1"
           "Don't waste your time.  We had two...", "1"
           "...", "..."

        :param data_bundle:
        :return: 处理后的 ``data_bundle``
        """
        if self.tag_map is not None:
            data_bundle = _granularize(data_bundle, self.tag_map)

        data_bundle = super().process(data_bundle)

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        data_bundle = YelpFullLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class YelpPolarityPipe(CLSBasePipe):
    r"""
    处理 **Yelp Review Polarity** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table:: 下面是使用YelpFullPipe处理后的DataSet所具备的field
        :header: "raw_words", "target", "words", "seq_len"

        "I got 'new' tires from them and within...", 0 ,"[7, 110, 22, 107, 22, 499, 59, 140, 3,...]", 160
        " Don't waste your time.  We had two dif... ", 0, "[277, 17, 278, 38, 30, 112, 24, 85, 27...", 40
        "...", ., "[...]", .

    :param lower: 是否对输入进行小写化。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw']`` 。``'raw'`` 表示使用空格作为切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param num_proc: 处理数据时使用的进程数目。
    """

    def __init__(self, lower: bool = False, tokenizer: str = 'spacy', num_proc: int=0):
        super().__init__(lower=lower, tokenizer=tokenizer, lang='en', num_proc=num_proc)

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        data_bundle = YelpPolarityLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class AGsNewsPipe(CLSBasePipe):
    r"""
    处理 **AG's News** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table:: 下面是使用AGsNewsPipe处理后的DataSet所具备的field
        :header: "raw_words", "target", "words", "seq_len"

        "I got 'new' tires from them and within...", 0 ,"[7, 110, 22, 107, 22, 499, 59, 140, 3,...]", 160
        " Don't waste your time.  We had two dif... ", 0, "[277, 17, 278, 38, 30, 112, 24, 85, 27...", 40
        "...", ., "[...]", .

    :param lower: 是否对输入进行小写化。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw']`` 。``'raw'`` 表示使用空格作为切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param num_proc: 处理数据时使用的进程数目。
    """

    def __init__(self, lower: bool = False, tokenizer: str = 'spacy', num_proc=0):
        super().__init__(lower=lower, tokenizer=tokenizer, lang='en', num_proc=num_proc)

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        data_bundle = AGsNewsLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class DBPediaPipe(CLSBasePipe):
    r"""
    处理 **DBPedia** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table:: 下面是使用DBPediaPipe处理后的DataSet所具备的field
        :header: "raw_words", "target", "words", "seq_len"

        "I got 'new' tires from them and within...", 0 ,"[7, 110, 22, 107, 22, 499, 59, 140, 3,...]", 160
        " Don't waste your time.  We had two dif... ", 0, "[277, 17, 278, 38, 30, 112, 24, 85, 27...", 40
        "...", ., "[...]", .

    :param lower: 是否对输入进行小写化。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw']`` 。``'raw'`` 表示使用空格作为切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param num_proc: 处理数据时使用的进程数目。
    """

    def __init__(self, lower: bool = False, tokenizer: str = 'spacy', num_proc: int=0):
        super().__init__(lower=lower, tokenizer=tokenizer, lang='en', num_proc=num_proc)

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        data_bundle = DBPediaLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class SSTPipe(CLSBasePipe):
    r"""
    处理 **SST** 的数据，处理之后， :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table:: 下面是使用SSTPipe处理后的DataSet所具备的field
        :header: "raw_words", "words", "target", "seq_len"

        "It 's a lovely film with lovely perfor...", 1, "[187, 6, 5, 132, 120, 70, 132, 188, 25...", 13
        "No one goes unindicted here , which is...", 0, "[191, 126, 192, 193, 194, 4, 195, 17, ...", 13
        "...", ., "[...]", .

    :param subtree: 是否将训练集、测试集和验证集数据展开为子树，扩充数据量。
    :param train_subtree: 是否将训练集通过子树扩展数据。
    :param lower: 是否对输入进行小写化。
    :param granularity: 支持 ``[2, 3, 5]`` 。若为 ``2`` ，则认为是二分类问题，将 **1、2** 归为一类， **4、5** 归为一类，
        丢掉 3；若为 ``3`` ，则认为是三分类问题，将 **1、2** 归为一类， **3** 归为一类， **4、5** 归为一类；若为 ``5`` ，则认为是五分类问题。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw']`` 。``'raw'`` 表示使用空格作为切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param num_proc: 处理数据时使用的进程数目。
    """
    def __init__(self, subtree: bool=False, train_subtree: bool=True, lower: bool=False, granularity: int=5, tokenizer: int='spacy', num_proc: int=0):
        super().__init__(tokenizer=tokenizer, lang='en', num_proc=num_proc)
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

    def process(self, data_bundle: DataBundle) -> DataBundle:
        r"""
        ``data_bunlde`` 中的 :class:`~fastNLP.core.DataSet` ` 应该至少拥有 ``raw_words`` 列，内容类似于：

        .. csv-table:: 下面是使用 SSTLoader 读取的 DataSet 所具备的 field
            :header: "raw_words"

            "(2 (3 (3 Effective) (2 but)) (1 (1 too-tepid)..."
            "(3 (3 (2 If) (3 (2 you) (3 (2 sometimes) ..."
            "..."

        :param data_bundle: 需要处理的 :class:`~fastNLP.io.DataBundle` 对象
        :return: 处理后的 ``data_bundle``
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

        # 根据granularity设置tag
        data_bundle = _granularize(data_bundle, tag_map=self.tag_map)

        data_bundle = super().process(data_bundle)

        return data_bundle

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        data_bundle = SSTLoader().load(paths)
        return self.process(data_bundle=data_bundle)


class SST2Pipe(CLSBasePipe):
    r"""
    处理 **SST-2** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table::
       :header: "raw_words", "target", "words", "seq_len"

       "it 's a charming and often affecting j... ", 1, "[19, 9, 6, 111, 5, 112, 113, 114, 3]", 9
       "unflinchingly bleak and desperate", 0, "[115, 116, 5, 117]", 4
       "...", "...", ., .

    :param lower: 是否对输入进行小写化。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw']`` 。``'raw'`` 表示使用空格作为切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param num_proc: 处理数据时使用的进程数目。
    """

    def __init__(self, lower=False, tokenizer='raw', num_proc=0):
        super().__init__(lower=lower, tokenizer=tokenizer, lang='en', num_proc=num_proc)

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        data_bundle = SST2Loader().load(paths)
        return self.process(data_bundle)


class IMDBPipe(CLSBasePipe):
    r"""
    处理 **IMDb** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table:: 输出 DataSet 的 field
       :header: "raw_words", "target", "words", "seq_len"

       "Bromwell High is a cartoon ... ", 0, "[3, 5, 6, 9, ...]", 20
       "Story of a man who has ...", 1, "[20, 43, 9, 10, ...]", 31
       "...", ., "[...]", .

    其中 ``raw_words`` 为 :class:`str` 类型，是原文； ``words`` 是转换为 index 的输入； ``target`` 是转换为 index 的目标值。
    ``words`` 列被设置为 input， ``target`` 列被设置为 target。

    :param lower: 是否对输入进行小写化。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw']`` 。``'raw'`` 表示使用空格作为切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param num_proc: 处理数据时使用的进程数目。
    """

    def __init__(self, lower: bool = False, tokenizer: str = 'spacy', num_proc=0):
        super().__init__(tokenizer=tokenizer, lang='en', num_proc=num_proc)
        self.lower = lower

    def process(self, data_bundle: DataBundle):
        r"""
        ``data_bunlde`` 中的 :class:`~fastNLP.core.DataSet` 应该具备如下的结构：有两个 field ， ``raw_words`` 和 ``target`` ，
        且均为 :class:`str` 类型。

        .. csv-table:: 输入DataSet的field
           :header: "raw_words", "target"

           "Bromwell High is a cartoon ... ", "pos"
           "Story of a man who has ...", "neg"
           "...", "..."

        :param DataBunlde data_bundle: 传入的DataBundle中的DataSet必须包含raw_words和target两个field，且raw_words列应该为str,
            target列应该为str。
        :return:  处理后的 ``data_bundle``
        """

        # 替换<br />
        def replace_br(raw_words):
            raw_words = raw_words.replace("<br />", ' ')
            return raw_words

        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(replace_br, field_name='raw_words', new_field_name='raw_words', num_proc=self.num_proc)

        data_bundle = super().process(data_bundle)

        return data_bundle

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        # 读取数据
        data_bundle = IMDBLoader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class ChnSentiCorpPipe(Pipe):
    r"""
    处理 **ChnSentiCorp** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容为：

    .. csv-table::
        :header: "raw_chars", "target", "chars", "seq_len"

        "這間酒店環境和服務態度亦算不錯,但房間空間太小~~", 1, "[2, 3, 4, 5, ...]", 31
        "<荐书> 推荐所有喜欢<红楼>...", 1, "[10, 21, ....]", 25
        "..."

    其中 ``chars`` , ``seq_len`` 是 input， ``target`` 是 target。

    :param bigrams: 是否增加一列 ``bigrams`` 。 ``bigrams`` 会对原文进行如下转化： ``['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]`` 。如果
        设置为 ``True`` ，返回的 `~fastNLP.core.DataSet` 将有一列名为 ``bigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('bigrams')`` 获取。
    :param trigrams: 是否增加一列 ``trigrams`` 。 ``trigrams`` 会对原文进行如下转化 ``['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]`` 。
        如果设置为 ``True`` ，返回的 `~fastNLP.core.DataSet` 将有一列名为 ``trigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('trigrams')`` 获取。
    :param num_proc: 处理数据时使用的进程数目。
    """

    def __init__(self, bigrams: bool=False, trigrams: bool=False, num_proc: int = 0):
        super().__init__()

        self.bigrams = bigrams
        self.trigrams = trigrams
        self.num_proc = num_proc

    def _tokenize(self, data_bundle):
        r"""
        将 DataSet 中的"复旦大学"拆分为 ["复", "旦", "大", "学"] . 未来可以通过扩展这个函数实现分词。

        :param data_bundle:
        :return:
        """
        data_bundle.apply_field(list, field_name='chars', new_field_name='chars')
        return data_bundle

    def process(self, data_bundle: DataBundle):
        r"""
        ``data_bunlde`` 中的 :class:`~fastNLP.core.DataSet` 应该具备如下的结构：

        .. csv-table::
            :header: "raw_chars", "target"

            "這間酒店環境和服務態度亦算不錯,但房間空間太小~~", "1"
            "<荐书> 推荐所有喜欢<红楼>...", "1"
            "..."

        :param data_bundle:
        :return:  处理后的 ``data_bundle``
        """
        _add_chars_field(data_bundle, lower=False)

        data_bundle = self._tokenize(data_bundle)

        input_field_names = ['chars']

        def bigrams(chars):
            return [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])]

        def trigrams(chars):
            return [c1 + c2 + c3 for c1, c2, c3 in
                    zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)]

        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(bigrams,field_name='chars', new_field_name='bigrams', num_proc=self.num_proc)
            input_field_names.append('bigrams')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(trigrams, field_name='chars', new_field_name='trigrams', num_proc=self.num_proc)
            input_field_names.append('trigrams')

        # index
        _indexize(data_bundle, input_field_names, 'target')

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len('chars')

        return data_bundle

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        # 读取数据
        data_bundle = ChnSentiCorpLoader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class THUCNewsPipe(CLSBasePipe):
    r"""
    处理 **THUCNews** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容为：

    .. csv-table::
        :header: "raw_chars", "target", "chars", "seq_len"

        "马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道...", 0, "[409, 1197, 2146, 213, ...]", 746
        "..."

    其中 ``chars`` , ``seq_len`` 是 input， ``target`` 是target

    :param bigrams: 是否增加一列 ``bigrams`` 。 ``bigrams`` 会对原文进行如下转化： ``['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]`` 。如果
        设置为 ``True`` ，返回的 `~fastNLP.core.DataSet` 将有一列名为 ``bigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('bigrams')`` 获取。
    :param trigrams: 是否增加一列 ``trigrams`` 。 ``trigrams`` 会对原文进行如下转化 ``['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]`` 。
        如果设置为 ``True`` ，返回的 `~fastNLP.core.DataSet` 将有一列名为 ``trigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('trigrams')`` 获取。
    :param num_proc: 处理数据时使用的进程数目。
    """

    def __init__(self, bigrams: int=False, trigrams: int=False, num_proc: int=0):
        super().__init__(num_proc=num_proc)

        self.bigrams = bigrams
        self.trigrams = trigrams

    def _chracter_split(self, sent):
        return list(sent)
        # return [w for w in sent]

    def _raw_split(self, sent):
        return sent.split()

    def _tokenize(self, data_bundle, field_name='words', new_field_name=None):
        new_field_name = new_field_name or field_name
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self._chracter_split, field_name=field_name, new_field_name=new_field_name, num_proc=self.num_proc)
        return data_bundle

    def process(self, data_bundle: DataBundle):
        r"""
        ``data_bunlde`` 中的 :class:`~fastNLP.core.DataSet` 应该具备如下的结构：

        .. csv-table::
            :header: "raw_words", "target"

            "马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 ... ", "体育"
            "...", "..."

        :param data_bundle:
        :return:  处理后的 ``data_bundle``
        """
        # 根据granularity设置tag
        tag_map = {'体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4, '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9}
        data_bundle = _granularize(data_bundle=data_bundle, tag_map=tag_map)

        # clean,lower

        # CWS(tokenize)
        data_bundle = self._tokenize(data_bundle=data_bundle, field_name='raw_chars', new_field_name='chars')

        input_field_names = ['chars']

        def bigrams(chars):
            return [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])]

        def trigrams(chars):
            return [c1 + c2 + c3 for c1, c2, c3 in
                    zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)]

        # n-grams
        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(bigrams, field_name='chars', new_field_name='bigrams', num_proc=self.num_proc)
            input_field_names.append('bigrams')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(trigrams, field_name='chars', new_field_name='trigrams', num_proc=self.num_proc)
            input_field_names.append('trigrams')

        # index
        data_bundle = _indexize(data_bundle=data_bundle, input_field_names='chars')

        # add length
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(field_name='chars', new_field_name='seq_len')

        return data_bundle

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        data_loader = THUCNewsLoader()  # 此处需要实例化一个data_loader，否则传入load()的参数为None
        data_bundle = data_loader.load(paths)
        data_bundle = self.process(data_bundle)
        return data_bundle


class WeiboSenti100kPipe(CLSBasePipe):
    r"""
    处理 **WeiboSenti100k** 的数据，处理之后 :class:`~fastNLP.core.DataSet` 中的内容为：

    .. csv-table::
        :header: "raw_chars", "target", "chars", "seq_len"

        "马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道...", 0, "[409, 1197, 2146, 213, ...]", 746
        "..."

    其中 ``chars`` , ``seq_len`` 是 input， ``target`` 是target

    :param bigrams: 是否增加一列 ``bigrams`` 。 ``bigrams`` 会对原文进行如下转化： ``['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]`` 。如果
        设置为 ``True`` ，返回的 `~fastNLP.core.DataSet` 将有一列名为 ``bigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('bigrams')`` 获取。
    :param trigrams: 是否增加一列 ``trigrams`` 。 ``trigrams`` 会对原文进行如下转化 ``['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]`` 。
        如果设置为 ``True`` ，返回的 `~fastNLP.core.DataSet` 将有一列名为 ``trigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('trigrams')`` 获取。
    :param num_proc: 处理数据时使用的进程数目。
    """

    def __init__(self, bigrams=False, trigrams=False, num_proc=0):
        super().__init__(num_proc=num_proc)

        self.bigrams = bigrams
        self.trigrams = trigrams

    def _chracter_split(self, sent):
        return list(sent)

    def _tokenize(self, data_bundle, field_name='words', new_field_name=None):
        new_field_name = new_field_name or field_name
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self._chracter_split, field_name=field_name,
                                new_field_name=new_field_name, num_proc=self.num_proc)
        return data_bundle

    def process(self, data_bundle: DataBundle):
        r"""
        ``data_bunlde`` 中的 :class:`~fastNLP.core.DataSet` 应该具备如下的结构：

        .. csv-table::
            :header: "raw_chars", "target"

            "六一出生的？好讽刺…… //@祭春姬:他爸爸是外星人吧 //@面孔小高:现在的孩子都怎么了 [怒][怒][怒]", "0"
            "...", "..."

        :param data_bundle:
        :return:  处理后的 ``data_bundle``
        """
        # clean,lower

        # CWS(tokenize)
        data_bundle = self._tokenize(data_bundle=data_bundle, field_name='raw_chars', new_field_name='chars')

        def bigrams(chars):
            return [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])]

        def trigrams(chars):
            return [c1 + c2 + c3 for c1, c2, c3 in
                    zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)]
        # n-grams
        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(bigrams, field_name='chars', new_field_name='bigrams', num_proc=self.num_proc)
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(trigrams, field_name='chars', new_field_name='trigrams', num_proc=self.num_proc)

        # index
        data_bundle = _indexize(data_bundle=data_bundle, input_field_names='chars')

        # add length
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(field_name='chars', new_field_name='seq_len')

        return data_bundle
    
    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        data_loader = WeiboSenti100kLoader()  # 此处需要实例化一个data_loader，否则传入load()的参数为None
        data_bundle = data_loader.load(paths)
        data_bundle = self.process(data_bundle)
        return data_bundle

class MRPipe(CLSBasePipe):
    """
    加载 **MR** 的数据。

    :param lower: 是否对输入进行小写化。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw']`` 。``'raw'`` 表示使用空格作为切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param num_proc: 处理数据时使用的进程数目。
    """
    def __init__(self, lower: bool = False, tokenizer: str = 'spacy', num_proc=0):
        super().__init__(tokenizer=tokenizer, lang='en', num_proc=num_proc)
        self.lower = lower

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        # 读取数据
        data_bundle = MRLoader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class R8Pipe(CLSBasePipe):
    """
    加载 **R8** 的数据。

    :param lower: 是否对输入进行小写化。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw']`` 。``'raw'`` 表示使用空格作为切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param num_proc: 处理数据时使用的进程数目。
    """
    def __init__(self, lower: bool = False, tokenizer: str = 'spacy', num_proc = 0):
        super().__init__(tokenizer=tokenizer, lang='en', num_proc=num_proc)
        self.lower = lower

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        # 读取数据
        data_bundle = R8Loader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class R52Pipe(CLSBasePipe):
    """
    加载 **R52** 的数据。

    :param lower: 是否对输入进行小写化。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw']`` 。``'raw'`` 表示使用空格作为切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param num_proc: 处理数据时使用的进程数目。
    """
    def __init__(self, lower: bool = False, tokenizer: str = 'spacy', num_proc: int = 0):
        super().__init__(tokenizer=tokenizer, lang='en', num_proc=num_proc)
        self.lower = lower

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        # 读取数据
        data_bundle = R52Loader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class OhsumedPipe(CLSBasePipe):
    """
    加载 **Ohsumed** 的数据。

    :param lower: 是否对输入进行小写化。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw']`` 。``'raw'`` 表示使用空格作为切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param num_proc: 处理数据时使用的进程数目。
    """
    def __init__(self, lower: bool = False, tokenizer: str = 'spacy', num_proc: int = 0):
        super().__init__(tokenizer=tokenizer, lang='en', num_proc=num_proc)
        self.lower = lower

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        # 读取数据
        data_bundle = OhsumedLoader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class NG20Pipe(CLSBasePipe):
    """
    加载 **NG20** 的数据。

    :param lower: 是否对输入进行小写化。
    :param tokenizer: 使用哪种 tokenize 方式将数据切成单词。支持 ``['spacy', 'raw']`` 。``'raw'`` 表示使用空格作为切分，``'spacy'`` 则使用 :mod:`spacy` 库进行分词。
    :param num_proc: 处理数据时使用的进程数目。
    """
    def __init__(self, lower: bool = False, tokenizer: str = 'spacy', num_proc: int = 0):
        super().__init__(tokenizer=tokenizer, lang='en', num_proc=num_proc)
        self.lower = lower

    def process_from_file(self, paths=None):
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        # 读取数据
        data_bundle = NG20Loader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle