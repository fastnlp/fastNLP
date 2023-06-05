r"""
.. todo::
    doc
"""

__all__ = [
    "Vocabulary",
    "VocabularyOption",
]

from collections import Counter
from functools import partial
from functools import wraps
from typing import List, Callable, Union

from fastNLP.core.dataset import DataSet
from fastNLP.core.utils.utils import Option
from fastNLP.core.utils.utils import _is_iterable
from .log import logger
import io


class VocabularyOption(Option):
    """

    """
    def __init__(self,
                 max_size=None,
                 min_freq=None,
                 padding='<pad>',
                 unknown='<unk>'):
        super().__init__(
            max_size=max_size,
            min_freq=min_freq,
            padding=padding,
            unknown=unknown
        )


def _check_build_vocab(func: Callable):
    r"""
    A decorator to make sure the indexing is built before used.

    :param func: 传入的callable函数

    """
    
    @wraps(func)  # to solve missing docstring
    def _wrapper(self, *args, **kwargs):
        if self._word2idx is None or self.rebuild is True:
            self.build_vocab()
        return func(self, *args, **kwargs)
    
    return _wrapper


def _check_build_status(func):
    r"""
    A decorator to check whether the vocabulary updates after the last build.

    :param func: 用户传入要修饰的callable函数

    """
    
    @wraps(func)  # to solve missing docstring
    def _wrapper(self, *args, **kwargs):
        if self.rebuild is False:
            self.rebuild = True
            if self.max_size is not None and len(self.word_count) >= self.max_size:
                logger.warning("Vocabulary has reached the max size {} when calling {} method. "
                            "Adding more words may cause unexpected behaviour of Vocabulary. ".format(
                    self.max_size, func.__name__))
        return func(self, *args, **kwargs)
    
    return _wrapper


class Vocabulary(object):
    r"""
    用于构建, 存储和使用 `str` 到 `int` 的一一映射::

        from fastNLP.core import Vocabulary
        vocab = Vocabulary()
        word_list = "this is a word list".split()
        # vocab更新自己的字典，输入为list列表
        vocab.update(word_list)
        vocab["word"] # str to int
        vocab.to_word(5) # int to str

    :param max_size: `Vocabulary` 的最大大小, 即能存储词的最大数量
        若为 ``None`` , 则不限制大小。
    :param min_freq: 能被记录下的词在文本中的最小出现频率, 应大于或等于 1。
        若小于该频率, 词语将被视为 `unknown`. 若为 ``None`` , 所有文本中的词都被记录。
    :param padding: padding的字符. 如果设置为 ``None`` ,
        则vocabulary中不考虑padding, 也不计入词表大小，为 ``None`` 的情况多在为 label 建立 Vocabulary 的情况。
    :param unknown: unknown的字符，所有未被记录的词在转为 :class:`int` 时将被视为 `unknown` 。
        如果设置为 ``None`` ,则 vocabulary 中不考虑 `unknown`, 也不计入词表大小。
        为 ``None`` 的情况多在为 labe l建立 Vocabulary 的情况
    """

    def __init__(self, max_size:int=None, min_freq:int=None, padding:str='<pad>', unknown:str='<unk>'):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word_count = Counter()
        self.unknown = unknown
        self.padding = padding
        self._word2idx = None
        self._idx2word = None
        self.rebuild = True
        #  用于承载不需要单独创建entry的词语，具体见from_dataset()方法
        self._no_create_word = Counter()

    @property
    @_check_build_vocab
    def word2idx(self):
        return self._word2idx

    @word2idx.setter
    def word2idx(self, value):
        self._word2idx = value

    @property
    @_check_build_vocab
    def idx2word(self):
        return self._idx2word

    @idx2word.setter
    def idx2word(self, value):
        self._word2idx = value

    @_check_build_status
    def update(self, word_lst: list, no_create_entry:bool=False):
        r"""
        依次增加序列中词在词典中的出现频率

        :param word_lst: 列表形式的词语，如 word_list=['I', 'am', 'a', 'Chinese']，列表中的每个词会计算出现频率并加入到词典中。
        :param no_create_entry: 如果词语来自于非训练集建议设置为 ``True`` 。
            
            * 如果为 ``True`` -- 则不会有这个词语创建一个单独的 entry ，它将一直被指向 ``<UNK>`` 的表示；
            * 如果为 ``False`` -- 为这个词创建一个单独的 entry。如果这个词来自于验证集或训练集，一般设置为True，如果来自于训练集一
              般设置为``False``；
              
            有以下两种情况: 如果新加入一个 word ，且 ``no_create_entry`` 为 ``True``，但这个词之前已经在 Vocabulary 中且并不是 
            ``no_create_entry`` 的，则还是会为这个词创建一个单独的 vector ; 如果 ``no_create_entry`` 为 ``False`` ，但这个词之
            前已经在 Vocabulary 中且并不是 ``no_create_entry的`` ，则这个词将认为是需要创建单独的 vector 的。

        """
        self._add_no_create_entry(word_lst, no_create_entry)
        self.word_count.update(word_lst)
        return self
    
    @_check_build_status
    def add(self, word:str, no_create_entry:bool=False):
        r"""
        增加一个新词在词典中的出现频率

        :param word: 要添加进字典的新词， ``word`` 为一个字符串
        :param no_create_entry: 如果词语来自于非训练集建议设置为 ``True`` 。
            
            * 如果为 ``True`` -- 则不会有这个词语创建一个单独的 entry ，它将一直被指向 ``<UNK>`` 的表示；
            * 如果为 ``False`` -- 为这个词创建一个单独的 entry。如果这个词来自于验证集或训练集，一般设置为 ``True`` ，如果来自于训练集一
              般设置为 ``False``；
              
            有以下两种情况: 如果新加入一个 word ，且 ``no_create_entry`` 为 ``True``，但这个词之前已经在 Vocabulary 中且并不是 
            ``no_create_entry`` 的，则还是会为这个词创建一个单独的 vector ; 如果 ``no_create_entry`` 为 ``False`` ，但这个词之
            前已经在 Vocabulary 中且并不是 ``no_create_entry的`` ，则这个词将认为是需要创建单独的 vector 的。

        """
        self._add_no_create_entry(word, no_create_entry)
        self.word_count[word] += 1
        return self
    
    def _add_no_create_entry(self, word:Union[str, List[str]], no_create_entry:bool):
        r"""
        在新加入word时，检查_no_create_word的设置。

        :param word: 要添加的新词或者是 :class:`List`类型的新词，如 word='I' 或者 word=['I', 'am', 'a', 'Chinese'] 均可
        :param no_create_entry: 如果词语来自于非训练集建议设置为 ``True`` 。
            
            * 如果为 ``True`` -- 则不会有这个词语创建一个单独的 entry ，它将一直被指向 ``<UNK>`` 的表示；
            * 如果为 ``False`` -- 为这个词创建一个单独的 entry。如果这个词来自于验证集或训练集，一般设置为 ``True`` ，如果来自于训练集一
              般设置为 ``False``；
    
        :return:

        """
        if isinstance(word, str) or not _is_iterable(word):
            word = [word]
        for w in word:
            if no_create_entry and self.word_count.get(w, 0) == self._no_create_word.get(w, 0):
                self._no_create_word[w] += 1
            elif not no_create_entry and w in self._no_create_word:
                self._no_create_word.pop(w)
    
    @_check_build_status
    def add_word(self, word:str, no_create_entry:bool=False):
        r"""
        增加一个新词在词典中的出现频率

        :param word: 要添加进字典的新词， ``word`` 为一个字符串
        :param no_create_entry: 如果词语来自于非训练集建议设置为 ``True`` 。
            
            * 如果为 ``True`` -- 则不会有这个词语创建一个单独的 entry ，它将一直被指向 ``<UNK>`` 的表示；
            * 如果为 ``False`` -- 为这个词创建一个单独的 entry。如果这个词来自于验证集或训练集，一般设置为 ``True`` ，如果来自于训练集一
              般设置为 ``False``；
              
            有以下两种情况: 如果新加入一个 word ，且 ``no_create_entry`` 为 ``True``，但这个词之前已经在 Vocabulary 中且并不是 
            ``no_create_entry`` 的，则还是会为这个词创建一个单独的 vector ; 如果 ``no_create_entry`` 为 ``False`` ，但这个词之
            前已经在 Vocabulary 中且并不是 ``no_create_entry的`` ，则这个词将认为是需要创建单独的 vector 的。

        """
        self.add(word, no_create_entry=no_create_entry)
    
    @_check_build_status
    def add_word_lst(self, word_lst: List[str], no_create_entry:bool=False):
        r"""
        依次增加序列中词在词典中的出现频率

        :param word_lst: 需要添加的新词的 list 序列，如 word_lst=['I', 'am', 'a', 'Chinese'] 。
        :param no_create_entry: 如果词语来自于非训练集建议设置为 ``True`` 。
            
            * 如果为 ``True`` -- 则不会有这个词语创建一个单独的 entry ，它将一直被指向 ``<UNK>`` 的表示；
            * 如果为 ``False`` -- 为这个词创建一个单独的 entry。如果这个词来自于验证集或训练集，一般设置为 ``True`` ，如果来自于训练集一
              般设置为 ``False``；
              
            有以下两种情况: 如果新加入一个 word ，且 ``no_create_entry`` 为 ``True``，但这个词之前已经在 Vocabulary 中且并不是 
            ``no_create_entry`` 的，则还是会为这个词创建一个单独的 vector ; 如果 ``no_create_entry`` 为 ``False`` ，但这个词之
            前已经在 Vocabulary 中且并不是 ``no_create_entry的`` ，则这个词将认为是需要创建单独的 vector 的。

        """
        self.update(word_lst, no_create_entry=no_create_entry)
        return self
    
    def build_vocab(self):
        r"""
        根据已经出现的词和出现频率构建词典。注意：重复构建可能会改变词典的大小，
        但已经记录在词典中的词，不会改变对应的 :class:`int`
        """
        if self._word2idx is None:
            self._word2idx = {}
            if self.padding is not None:
                self._word2idx[self.padding] = len(self._word2idx)
            if (self.unknown is not None) and (self.unknown != self.padding):
                self._word2idx[self.unknown] = len(self._word2idx)
        
        max_size = min(self.max_size, len(self.word_count)) if self.max_size else None
        words = self.word_count.most_common(max_size)
        if self.min_freq is not None:
            words = filter(lambda kv: kv[1] >= self.min_freq, words)
        if self._word2idx is not None:
            words = filter(lambda kv: kv[0] not in self._word2idx, words)
        start_idx = len(self._word2idx)
        self._word2idx.update({w: i + start_idx for i, (w, _) in enumerate(words)})
        self.build_reverse_vocab()
        self.rebuild = False
        return self
    
    def build_reverse_vocab(self):
        r"""
        基于 `word to index` dict, 构建 `index to word` dict.

        """
        self._idx2word = {i: w for w, i in self._word2idx.items()}
        return self
    
    @_check_build_vocab
    def __len__(self):
        return len(self._word2idx)
    
    @_check_build_vocab
    def __contains__(self, item:str):
        r"""
        检查词是否被记录

        :param item: the word
        :return: True or False
        """
        return item in self._word2idx
    
    def has_word(self, w:str):
        r"""
        检查词是否被记录::

            has_abc = vocab.has_word('abc')
            # equals to
            has_abc = 'abc' in vocab

        :param item: 输入的str类型的词
        :return: ``True`` or ``False``
        """
        return self.__contains__(w)
    
    @_check_build_vocab
    def __getitem__(self, w):
        r"""
        支持从字典中直接得到词语的index，例如::

            vocab[w]
        """
        if w in self._word2idx:
            return self._word2idx[w]
        if self.unknown is not None:
            return self._word2idx[self.unknown]
        else:
            raise ValueError("word `{}` not in vocabulary".format(w))
    
    @_check_build_vocab
    def index_dataset(self, *datasets, field_name:Union[List, str], new_field_name:Union[List, str, None]=None):
        r"""
        将 ``DataSet`` 中对应 field 的词转为数字，例如::

            # remember to use `field_name`
            vocab.index_dataset(train_data, dev_data, test_data, field_name='words')

        :param datasets: 其类型为 :class:`~fastNLP.core.dataset.DataSet` 或者 :class:`List` [ :class:`~fastNLP.core.dataset.DataSet` ]，
            即需要处理的一个或多个数据集
        :param field_name: 需要转为 index 的 field, 若有多个 DataSet, 每个 DataSet 都必须有此 field.
            目前支持 :class:`str` , :class:`List` [ :class:`str` ]
        :param new_field_name: 保存结果的 field_name。 若为 ``None`` , 将覆盖原 field。
        """
        
        def index_instance(field):
            r"""
            有几种情况, str, 1d-list, 2d-list
            :param ins:
            :return:
            """
            if isinstance(field, str) or not _is_iterable(field):
                return self.to_index(field)
            else:
                if isinstance(field[0], str) or not _is_iterable(field[0]):
                    return [self.to_index(w) for w in field]
                else:
                    if not isinstance(field[0][0], str) and _is_iterable(field[0][0]):
                        raise RuntimeError("Only support field with 2 dimensions.")
                    return [[self.to_index(c) for c in w] for w in field]
        
        new_field_name = new_field_name or field_name
        
        if type(new_field_name) == type(field_name):
            if isinstance(new_field_name, list):
                assert len(new_field_name) == len(field_name), "new_field_name should have same number elements with " \
                                                               "field_name."
            elif isinstance(new_field_name, str):
                field_name = [field_name]
                new_field_name = [new_field_name]
            else:
                raise TypeError("field_name and new_field_name can only be str or List[str].")
        
        for idx, dataset in enumerate(datasets):
            if isinstance(dataset, DataSet):
                ds_lst = [dataset]
            elif _is_iterable(dataset):
                ds_lst = list(dataset)
            else:
                raise TypeError(f"Only DataSet type is allowed, instead of {type(dataset)}.")
            try:
                for ds in ds_lst:
                    for f_n, n_f_n in zip(field_name, new_field_name):
                        ds.apply_field(index_instance, field_name=f_n, new_field_name=n_f_n, progress_bar=None)
            except Exception as e:
                logger.error("When processing the `{}` dataset, the following error occurred.".format(idx))
                raise e
        return self
    
    @property
    def _no_create_word_length(self):
        return len(self._no_create_word)
    
    def from_dataset(self, *datasets, field_name:Union[str,List[str]], no_create_entry_dataset=None):
        r"""
        使用dataset的对应field中词构建词典::

            # remember to use `field_name`
            vocab.from_dataset(train_data1, train_data2, field_name='words', no_create_entry_dataset=[test_data1, test_data2])

        :param datasets: 其类型为 :class:`~fastNLP.core.dataset.DataSet` 或者 List[:class:`~fastNLP.core.dataset.DataSet`]。
        :param field_name: 构建词典所使用的 field(s), 支持一个或多个 field，若有多个 DataSet, 每个 DataSet 都必须有这些 field.
            目前支持的field结构: ``str`` , ``List[str]``
        :param no_create_entry_dataset: 可以传入 :class:`~fastNLP.core.dataset.DataSet`, :class:`List` [ :class:`~fastNLP.core.dataset.DataSet` ] 或者 
            ``None`` （默认），建议直接将非训练数据都传入到这个参数。该选项用于接下来的模型会使用预训练的 embedding （包括 ``glove``, ``word2vec`` ,
            ``elmo`` 与 ``bert`` ）且会 finetune 的情况。如果仅使用来自于训练集的数据建立词表，会导致测试集与验证集中的数据无法充分利用到来自于预训练
            embedding 的信息，所以在建立词表的时候将测试集与验证集考虑进来会使得最终的结果更好。
            如果一个词出现在了训练集中，但是没在预训练模型中， embedding 会为它用 ``<UNK>`` 初始化；但如果它是单独的一个 vector ，并且 finetune embedding
            的话，这个词在更新之后可能会有更好的表示；而如果这个词仅出现在了验证集或者测试集中，那么就不能为它们单独建立 vector，而应该让它指向 ``<UNK>`` 这个
            vector 的值。所以只位于 ``no_create_entry_dataset`` 中的 token 将首先从预训练的词表中寻找它的表示，如果找到了，就使用该表示; 如果没有找到，则认
            为该词的表示应该为 ``<UNK>`` 的表示。
        :return: Vocabulary 自身

        """
        if isinstance(field_name, str):
            field_name = [field_name]
        elif not isinstance(field_name, list):
            raise TypeError('invalid argument field_name: {}'.format(field_name))
        
        def construct_vocab(ins, no_create_entry=False):
            for fn in field_name:
                field = ins[fn]
                 # 如果 field 为空或者 None， 那么直接跳过即可。
                if field is None or (hasattr(field, "__len__") and len(field) == 0):
                    logger.warning(f"instance: {ins} has null field. Skip now!") 
                    continue
                if isinstance(field, str) or not _is_iterable(field):
                    self.add_word(field, no_create_entry=no_create_entry)
                else:
                    if isinstance(field[0], str) or not _is_iterable(field[0]):
                        for word in field:
                            self.add_word(word, no_create_entry=no_create_entry)
                    else:
                        if not isinstance(field[0][0], str) and _is_iterable(field[0][0]):
                            raise RuntimeError("Only support field with 2 dimensions.")
                        for words in field:
                            for word in words:
                                self.add_word(word, no_create_entry=no_create_entry)
        
        for idx, dataset in enumerate(datasets):
            if isinstance(dataset, DataSet):
                ds_lst = [dataset]
            elif _is_iterable(dataset):
                ds_lst = list(dataset)
            else:
                raise TypeError(f"Only DataSet type is allowed, instead of {type(dataset)}.")

            try:
                for ds in ds_lst:
                    ds.apply(construct_vocab, progress_bar=None)
            except BaseException as e:
                logger.error("When processing the `{}` dataset, the following error occurred:".format(idx))
                raise e
        
        if no_create_entry_dataset is not None:
            partial_construct_vocab = partial(construct_vocab, no_create_entry=True)
            if isinstance(no_create_entry_dataset, DataSet):
                no_create_entry_dataset.apply(partial_construct_vocab, progress_bar=None)
            elif isinstance(no_create_entry_dataset, list):
                for dataset in no_create_entry_dataset:
                    if not isinstance(dataset, DataSet):
                        raise TypeError("Only DataSet type is allowed.")
                    dataset.apply(partial_construct_vocab, progress_bar=None)
        return self
    
    def _is_word_no_create_entry(self, word:str):
        r"""
        判断当前的word是否是不需要创建entry的，具体参见from_dataset的说明

        :param word: 输入的str类型的词语
        :return: bool值的判断结果
        """
        return word in self._no_create_word
    
    def to_index(self, w:str):
        r"""
        将词转为数字。 若词不在词典中被记录, 将视为 `unknown`, 若 ``unknown=None`` , 将抛出 ``ValueError`` ::

            index = vocab.to_index('abc')
            # equals to
            index = vocab['abc']

        :param w: 需要输入的词语
        :return: 词语 ``w`` 对应的 :class:`int`类型的 index
        """
        return self.__getitem__(w)
    
    @property
    @_check_build_vocab
    def unknown_idx(self):
        r"""
        获得 ``unknown`` 对应的数字.
        """
        if self.unknown is None:
            return None
        return self._word2idx[self.unknown]
    
    @property
    @_check_build_vocab
    def padding_idx(self):
        r"""
        获得 ``padding`` 对应的数字
        """
        if self.padding is None:
            return None
        return self._word2idx[self.padding]
    
    @_check_build_vocab
    def to_word(self, idx: int):
        r"""
        给定一个数字, 将其转为对应的词.

        :param idx:
        :return: ``idx`` 对应的词
        """
        return self._idx2word[idx]
    
    def clear(self):
        r"""
        删除 :class:Vocabulary`` 中的词表数据。相当于重新初始化一下。

        :return: 自身
        """
        self.word_count.clear()
        self._word2idx = None
        self._idx2word = None
        self.rebuild = True
        self._no_create_word.clear()
        return self
    
    def __getstate__(self):
        r"""
        用来从 pickle 中加载 data

        """
        len(self)  # make sure vocab has been built
        state = self.__dict__.copy()
        # no need to pickle _idx2word as it can be constructed from _word2idx
        del state['_idx2word']
        return state
    
    def __setstate__(self, state):
        r"""
        支持 pickle 的保存，保存到 pickle 的 data state

        """
        self.__dict__.update(state)
        self.build_reverse_vocab()
    
    def __repr__(self):
        return "Vocabulary({}...)".format(list(self.word_count.keys())[:5])
    
    @_check_build_vocab
    def __iter__(self):
        # 依次(word1, 0), (word1, 1)
        for index in range(len(self._word2idx)):
            yield self.to_word(index), index

    def save(self, filepath: Union[str, io.StringIO]):
        r"""
        保存当前词表。

        :param filepath: 词表储存路径
        """
        if isinstance(filepath, io.IOBase):
            assert filepath.writable()
            f = filepath
        elif isinstance(filepath, str):
            try:
                f = open(filepath, 'w', encoding='utf-8')
            except Exception as e:
                raise e
        else:
            raise TypeError("Illegal `path`.")

        f.write(f'max_size\t{self.max_size}\n')
        f.write(f'min_freq\t{self.min_freq}\n')
        f.write(f'unknown\t{self.unknown}\n')
        f.write(f'padding\t{self.padding}\n')
        f.write(f'rebuild\t{self.rebuild}\n')
        f.write('\n')
        # idx: 如果idx为-2, 说明还没有进行build; 如果idx为-1，说明该词未编入
        # no_create_entry: 如果为1，说明该词是no_create_entry; 0 otherwise
        # word \t count \t idx \t no_create_entry \n
        idx = -2
        for word, count in self.word_count.items():
            if self._word2idx is not None:
                idx = self._word2idx.get(word, -1)
            is_no_create_entry = int(self._is_word_no_create_entry(word))
            f.write(f'{word}\t{count}\t{idx}\t{is_no_create_entry}\n')
        if isinstance(filepath, str):  # 如果是file的话就关闭
            f.close()

    @staticmethod
    def load(filepath: Union[str,io.StringIO]):
        r"""
        从文件路径中加载数据

        :param filepath: 词表的读取路径
        :return: 读取的 :class:`Vocabulary`
        """
        if isinstance(filepath, io.IOBase):
            assert filepath.writable()
            f = filepath
        elif isinstance(filepath, str):
            try:
                f = open(filepath, 'r', encoding='utf-8')
            except Exception as e:
                raise e
        else:
            raise TypeError("Illegal `path`.")

        vocab = Vocabulary()
        for line in f:
            line = line.strip('\n')
            if line:
                name, value = line.split()
                if name in ('max_size', 'min_freq'):
                    value = int(value) if value!='None' else None
                    setattr(vocab, name, value)
                elif name in ('unknown', 'padding'):
                    value = value if value!='None' else None
                    setattr(vocab, name, value)
                elif name == 'rebuild':
                    vocab.rebuild = True if value=='True' else False
            else:
                break
        word_counter = {}
        no_create_entry_counter = {}
        word2idx = {}
        for line in f:
            line = line.strip('\n')
            if line:
                parts = line.split('\t')
                word,count,idx,no_create_entry = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
                if idx >= 0:
                    word2idx[word] = idx
                word_counter[word] = count
                if no_create_entry:
                    no_create_entry_counter[word] = count

        word_counter = Counter(word_counter)
        no_create_entry_counter = Counter(no_create_entry_counter)
        if len(word2idx)>0:
            if vocab.padding:
                word2idx[vocab.padding] = 0
            if vocab.unknown:
                word2idx[vocab.unknown] = 1 if vocab.padding else 0
            idx2word = {value:key for key,value in word2idx.items()}

        vocab.word_count = word_counter
        vocab._no_create_word = no_create_entry_counter
        if word2idx:
            vocab._word2idx = word2idx
            vocab._idx2word = idx2word
        if isinstance(filepath, str):  # 如果是file的话就关闭
            f.close()
        return vocab
