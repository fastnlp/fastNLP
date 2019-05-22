__all__ = [
    "Vocabulary"
]

from functools import wraps
from collections import Counter

from .dataset import DataSet


def _check_build_vocab(func):
    """A decorator to make sure the indexing is built before used.

    """
    
    @wraps(func)  # to solve missing docstring
    def _wrapper(self, *args, **kwargs):
        if self.word2idx is None or self.rebuild is True:
            self.build_vocab()
        return func(self, *args, **kwargs)
    
    return _wrapper


def _check_build_status(func):
    """A decorator to check whether the vocabulary updates after the last build.

    """
    
    @wraps(func)  # to solve missing docstring
    def _wrapper(self, *args, **kwargs):
        if self.rebuild is False:
            self.rebuild = True
            if self.max_size is not None and len(self.word_count) >= self.max_size:
                print("[Warning] Vocabulary has reached the max size {} when calling {} method. "
                      "Adding more words may cause unexpected behaviour of Vocabulary. ".format(
                    self.max_size, func.__name__))
        return func(self, *args, **kwargs)
    
    return _wrapper


class Vocabulary(object):
    """
    别名：:class:`fastNLP.Vocabulary` :class:`fastNLP.core.vocabulary.Vocabulary`
    
    用于构建, 存储和使用 `str` 到 `int` 的一一映射::

        vocab = Vocabulary()
        word_list = "this is a word list".split()
        vocab.update(word_list)
        vocab["word"] # str to int
        vocab.to_word(5) # int to str

    :param int max_size: `Vocabulary` 的最大大小, 即能存储词的最大数量
        若为 ``None`` , 则不限制大小. Default: ``None``
    :param int min_freq: 能被记录下的词在文本中的最小出现频率, 应大于或等于 1.
        若小于该频率, 词语将被视为 `unknown`. 若为 ``None`` , 所有文本中的词都被记录. Default: ``None``
    :param str optional padding: padding的字符. 如果设置为 ``None`` ,
        则vocabulary中不考虑padding, 也不计入词表大小，为 ``None`` 的情况多在为label建立Vocabulary的情况.
        Default: '<pad>'
    :param str optional unknown: unknown的字符，所有未被记录的词在转为 `int` 时将被视为unknown.
        如果设置为 ``None`` ,则vocabulary中不考虑unknow, 也不计入词表大小.
        为 ``None`` 的情况多在为label建立Vocabulary的情况.
        Default: '<unk>'
    """
    
    def __init__(self, max_size=None, min_freq=None, padding='<pad>', unknown='<unk>'):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word_count = Counter()
        self.unknown = unknown
        self.padding = padding
        self.word2idx = None
        self.idx2word = None
        self.rebuild = True
    
    @_check_build_status
    def update(self, word_lst):
        """依次增加序列中词在词典中的出现频率

        :param list word_lst: a list of strings
        """
        self.word_count.update(word_lst)
    
    @_check_build_status
    def add(self, word):
        """
        增加一个新词在词典中的出现频率

        :param str word: 新词
        """
        self.word_count[word] += 1
    
    @_check_build_status
    def add_word(self, word):
        """
        增加一个新词在词典中的出现频率

        :param str word: 新词
        """
        self.add(word)
    
    @_check_build_status
    def add_word_lst(self, word_lst):
        """
        依次增加序列中词在词典中的出现频率

        :param list[str] word_lst: 词的序列
        """
        self.update(word_lst)
    
    def build_vocab(self):
        """
        根据已经出现的词和出现频率构建词典. 注意: 重复构建可能会改变词典的大小,
        但已经记录在词典中的词, 不会改变对应的 `int`

        """
        if self.word2idx is None:
            self.word2idx = {}
        if self.padding is not None:
            self.word2idx[self.padding] = len(self.word2idx)
        if self.unknown is not None:
            self.word2idx[self.unknown] = len(self.word2idx)
        
        max_size = min(self.max_size, len(self.word_count)) if self.max_size else None
        words = self.word_count.most_common(max_size)
        if self.min_freq is not None:
            words = filter(lambda kv: kv[1] >= self.min_freq, words)
        if self.word2idx is not None:
            words = filter(lambda kv: kv[0] not in self.word2idx, words)
        start_idx = len(self.word2idx)
        self.word2idx.update({w: i + start_idx for i, (w, _) in enumerate(words)})
        self.build_reverse_vocab()
        self.rebuild = False
    
    def build_reverse_vocab(self):
        """
        基于 "word to index" dict, 构建 "index to word" dict.

        """
        self.idx2word = {i: w for w, i in self.word2idx.items()}
    
    @_check_build_vocab
    def __len__(self):
        return len(self.word2idx)
    
    @_check_build_vocab
    def __contains__(self, item):
        """
        检查词是否被记录

        :param item: the word
        :return: True or False
        """
        return item in self.word2idx
    
    def has_word(self, w):
        """
        检查词是否被记录::

            has_abc = vocab.has_word('abc')
            # equals to
            has_abc = 'abc' in vocab

        :param item: the word
        :return: ``True`` or ``False``
        """
        return self.__contains__(w)
    
    @_check_build_vocab
    def __getitem__(self, w):
        """
        To support usage like::

            vocab[w]
        """
        if w in self.word2idx:
            return self.word2idx[w]
        if self.unknown is not None:
            return self.word2idx[self.unknown]
        else:
            raise ValueError("word {} not in vocabulary".format(w))
    
    @_check_build_vocab
    def index_dataset(self, *datasets, field_name, new_field_name=None):
        """
        将DataSet中对应field的词转为数字，Example::

            # remember to use `field_name`
            vocab.index_dataset(train_data, dev_data, test_data, field_name='words')

        :param datasets: 需要转index的 class:`~fastNLP.DataSet` , 支持一个或多个（list）
        :param str field_name: 需要转index的field, 若有多个 DataSet, 每个DataSet都必须有此 field.
            目前仅支持 ``str`` , ``list(str)`` , ``list(list(str))``
        :param str new_field_name: 保存结果的field_name. 若为 ``None`` , 将覆盖原field.
            Default: ``None``
        """
        
        def index_instance(ins):
            """
            有几种情况, str, 1d-list, 2d-list
            :param ins:
            :return:
            """
            field = ins[field_name]
            if isinstance(field, str):
                return self.to_index(field)
            elif isinstance(field, list):
                if not isinstance(field[0], list):
                    return [self.to_index(w) for w in field]
                else:
                    if isinstance(field[0][0], list):
                        raise RuntimeError("Only support field with 2 dimensions.")
                    return [[self.to_index(c) for c in w] for w in field]
        
        if new_field_name is None:
            new_field_name = field_name
        for idx, dataset in enumerate(datasets):
            if isinstance(dataset, DataSet):
                try:
                    dataset.apply(index_instance, new_field_name=new_field_name)
                except Exception as e:
                    print("When processing the `{}` dataset, the following error occurred.".format(idx))
                    raise e
            else:
                raise RuntimeError("Only DataSet type is allowed.")
    
    def from_dataset(self, *datasets, field_name):
        """
        使用dataset的对应field中词构建词典::

            # remember to use `field_name`
            vocab.from_dataset(train_data1, train_data2, field_name='words')

        :param datasets: 需要转index的 class:`~fastNLP.DataSet` , 支持一个或多个（list）
        :param field_name: 可为 ``str`` 或 ``list(str)`` .
            构建词典所使用的 field(s), 支持一个或多个field
            若有多个 DataSet, 每个DataSet都必须有这些field.
            目前仅支持的field结构: ``str`` , ``list(str)`` , ``list(list(str))``
        :return self:
        """
        if isinstance(field_name, str):
            field_name = [field_name]
        elif not isinstance(field_name, list):
            raise TypeError('invalid argument field_name: {}'.format(field_name))
        
        def construct_vocab(ins):
            for fn in field_name:
                field = ins[fn]
                if isinstance(field, str):
                    self.add_word(field)
                elif isinstance(field, list):
                    if not isinstance(field[0], list):
                        self.add_word_lst(field)
                    else:
                        if isinstance(field[0][0], list):
                            raise RuntimeError("Only support field with 2 dimensions.")
                        [self.add_word_lst(w) for w in field]
        
        for idx, dataset in enumerate(datasets):
            if isinstance(dataset, DataSet):
                try:
                    dataset.apply(construct_vocab)
                except Exception as e:
                    print("When processing the `{}` dataset, the following error occurred.".format(idx))
                    raise e
            else:
                raise RuntimeError("Only DataSet type is allowed.")
        return self
    
    def to_index(self, w):
        """
        将词转为数字. 若词不再词典中被记录, 将视为 unknown, 若 ``unknown=None`` , 将抛出
        ``ValueError``::

            index = vocab.to_index('abc')
            # equals to
            index = vocab['abc']

        :param str w: a word
        :return int index: the number
        """
        return self.__getitem__(w)
    
    @property
    @_check_build_vocab
    def unknown_idx(self):
        """
        unknown 对应的数字.
        """
        if self.unknown is None:
            return None
        return self.word2idx[self.unknown]
    
    @property
    @_check_build_vocab
    def padding_idx(self):
        """
        padding 对应的数字
        """
        if self.padding is None:
            return None
        return self.word2idx[self.padding]
    
    @_check_build_vocab
    def to_word(self, idx):
        """
        给定一个数字, 将其转为对应的词.

        :param int idx: the index
        :return str word: the word
        """
        return self.idx2word[idx]
    
    def clear(self):
        """
        删除Vocabulary中的词表数据。相当于重新初始化一下。

        :return:
        """
        self.word_count.clear()
        self.word2idx = None
        self.idx2word = None
        self.rebuild = True
    
    def __getstate__(self):
        """Use to prepare data for pickle.

        """
        len(self)  # make sure vocab has been built
        state = self.__dict__.copy()
        # no need to pickle idx2word as it can be constructed from word2idx
        del state['idx2word']
        return state
    
    def __setstate__(self, state):
        """Use to restore state from pickle.

        """
        self.__dict__.update(state)
        self.build_reverse_vocab()
    
    def __repr__(self):
        return "Vocabulary({}...)".format(list(self.word_count.keys())[:5])
    
    def __iter__(self):
        return iter(list(self.word_count.keys()))
