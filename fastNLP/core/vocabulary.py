from collections import Counter
import os
# 问题：fastNLP虽然已经提供了split函数，可以将数据集划分成训练集和测试机，但一般网上用作训练的标准集都已经提前划分好了训练集和测试机，
# 而使用split将数据集进行随机划分还引来了一个问题：
#       因为每次都是随机划分，导致每次的字典都不一样，保存好模型下次再载入进行测试时，因为字典不同导致结果差异非常大。
#
# 解决方法：在Vocabulary增加一个字典保存函数和一个字典读取函数，而不是每次都生成一个新字典，同时减少下次运行的成本，第一次使用save_vocab()
# 生成字典后，下次可以直接使用load_vocab()载入的字典。
#测试：在test文件夹下有test_vocabulary进行测试
def check_build_vocab(func):
    """A decorator to make sure the indexing is built before used.

    """

    def _wrapper(self, *args, **kwargs):
        if self.word2idx is None or self.rebuild is True:
            self.build_vocab()
        return func(self, *args, **kwargs)

    return _wrapper


def check_build_status(func):
    """A decorator to check whether the vocabulary updates after the last build.

    """

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
    """Use for word and index one to one mapping

    Example::

        vocab = Vocabulary()
        word_list = "this is a word list".split()
        vocab.update(word_list)
        vocab["word"]
        vocab.to_word(5)
    """

    def __init__(self, max_size=None, min_freq=None, unknown='<unk>', padding='<pad>',write_path='vocabulary',write_voc=True):
        """
        :param int max_size: set the max number of words in Vocabulary. Default: None
        :param int min_freq: set the min occur frequency of words in Vocabulary. Default: None
        """
        self.max_size = max_size
        self.min_freq = min_freq
        self.word_count = Counter()
        self.unknown = unknown
        self.padding = padding
        self.word2idx = None
        self.idx2word = None
        self.rebuild = True
        self.write_path=write_path
        self.write_voc=write_voc

    @check_build_status
    def update(self, word_lst):
        """Add a list of words into the vocabulary.

        :param list word_lst: a list of strings
        """
        self.word_count.update(word_lst)

    @check_build_status
    def add(self, word):
        """Add a single word into the vocabulary.

        :param str word: a word or token.
        """
        self.word_count[word] += 1

    @check_build_status
    def add_word(self, word):
        """Add a single word into the vocabulary.

        :param str word: a word or token.
        """
        self.add(word)

    @check_build_status
    def add_word_lst(self, word_lst):
        """Add a list of words into the vocabulary.

        :param list word_lst: a list of strings
        """
        self.update(word_lst)

    def build_vocab(self):
        """Build 'word to index' dict, and filter the word using `max_size` and `min_freq`.

        """

        self.word2idx = {}
        if self.padding is not None:
            self.word2idx[self.padding] = 0
        if self.unknown is not None:
            self.word2idx[self.unknown] = 1

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
        print('self.word2idx', self.word2idx)


    #新增功能，保存字典
    def save_vocab(self):
        if self.write_voc:
            # 保存已经处理好的字典
            file = os.path.join(self.write_path, 'vocab.data')
            with open(file, 'w', encoding='utf8') as f:
                for key in self.word2idx:
                    f.write(key + ';;' + str(self.word2idx[key]) + '\n')
        print('sucessfully save!')
    # 新增功能，载入字典
    def load_vocab(self):
        #载入已经处理好的字典
        file = os.path.join(self.write_path, 'vocab.data')
        with open(file, 'r', encoding='utf8') as f:
            context = f.readlines()
        word2id = {}
        for ix, vocab in enumerate(context):
            word2id[vocab.split(';;')[0]] = ix + 2
        self.word2idx=word2id
        print('sucessfully load!')
        return word2id


    def build_reverse_vocab(self):
        """Build 'index to word' dict based on 'word to index' dict.

        """
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    @check_build_vocab
    def __len__(self):
        return len(self.word2idx)

    @check_build_vocab
    def __contains__(self, item):
        """Check if a word in vocabulary.

        :param item: the word
        :return: True or False
        """
        return item in self.word2idx

    def has_word(self, w):
        return self.__contains__(w)

    @check_build_vocab
    def __getitem__(self, w):
        """To support usage like::

            vocab[w]
        """
        if w in self.word2idx:
            return self.word2idx[w]
        if self.unknown is not None:
            return self.word2idx[self.unknown]
        else:
            raise ValueError("word {} not in vocabulary".format(w))

    def to_index(self, w):
        """ Turn a word to an index.
            If w is not in Vocabulary, return the unknown label.

        :param str w:
        """
        return self.__getitem__(w)

    @property
    @check_build_vocab
    def unknown_idx(self):
        if self.unknown is None:
            return None
        return self.word2idx[self.unknown]

    @property
    @check_build_vocab
    def padding_idx(self):
        if self.padding is None:
            return None
        return self.word2idx[self.padding]

    @check_build_vocab
    def to_word(self, idx):
        """given a word's index, return the word itself

        :param int idx: the index
        :return str word: the indexed word
        """
        return self.idx2word[idx]

    def __getstate__(self):
        """Use to prepare data for pickle.

        """
        state = self.__dict__.copy()
        # no need to pickle idx2word as it can be constructed from word2idx
        del state['idx2word']
        return state

    def __setstate__(self, state):
        """Use to restore state from pickle.

        """
        self.__dict__.update(state)
        self.build_reverse_vocab()
