from collections import Counter
from copy import deepcopy

DEFAULT_PADDING_LABEL = '<pad>'  # dict index = 0
DEFAULT_UNKNOWN_LABEL = '<unk>'  # dict index = 1

DEFAULT_WORD_TO_INDEX = {DEFAULT_PADDING_LABEL: 0, DEFAULT_UNKNOWN_LABEL: 1}


def isiterable(p_object):
    try:
        _ = iter(p_object)
    except TypeError:
        return False
    return True


def check_build_vocab(func):
    def _wrapper(self, *args, **kwargs):
        if self.word2idx is None:
            self.build_vocab()
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

    def __init__(self, need_default=True, max_size=None, min_freq=None):
        """
        :param bool need_default: set if the Vocabulary has default labels reserved for sequences. Default: True.
        :param int max_size: set the max number of words in Vocabulary. Default: None
        :param int min_freq: set the min occur frequency of words in Vocabulary. Default: None
        """
        self.max_size = max_size
        self.min_freq = min_freq
        self.word_count = Counter()
        self.has_default = need_default
        if self.has_default:
            self.padding_label = DEFAULT_PADDING_LABEL
            self.unknown_label = DEFAULT_UNKNOWN_LABEL
        else:
            self.padding_label = None
            self.unknown_label = None
        self.word2idx = None
        self.idx2word = None

    def update(self, word_lst):
        """Add a list of words into the vocabulary.

        :param list word_lst: a list of strings
        """
        self.word_count.update(word_lst)

    def add(self, word):
        """Add a single word into the vocabulary.

        :param str word: a word or token.
        """
        self.word_count[word] += 1

    def add_word(self, word):
        """Add a single word into the vocabulary.

        :param str word: a word or token.
        """
        self.add(word)

    def add_word_lst(self, word_lst):
        """Add a list of words into the vocabulary.

        :param list word_lst: a list of strings
        """
        self.update(word_lst)

    def build_vocab(self):
        """Build 'word to index' dict, and filter the word using `max_size` and `min_freq`.

        """
        if self.has_default:
            self.word2idx = deepcopy(DEFAULT_WORD_TO_INDEX)
            self.word2idx[self.unknown_label] = self.word2idx.pop(DEFAULT_UNKNOWN_LABEL)
            self.word2idx[self.padding_label] = self.word2idx.pop(DEFAULT_PADDING_LABEL)
        else:
            self.word2idx = {}

        max_size = min(self.max_size, len(self.word_count)) if self.max_size else None
        words = self.word_count.most_common(max_size)
        if self.min_freq is not None:
            words = filter(lambda kv: kv[1] >= self.min_freq, words)
        start_idx = len(self.word2idx)
        self.word2idx.update({w: i + start_idx for i, (w, _) in enumerate(words)})
        self.build_reverse_vocab()

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
        elif self.has_default:
            return self.word2idx[self.unknown_label]
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
        if self.unknown_label is None:
            return None
        return self.word2idx[self.unknown_label]

    def __setattr__(self, name, val):
        self.__dict__[name] = val
        if name in ["unknown_label", "padding_label"]:
            self.word2idx = None

    @property
    @check_build_vocab
    def padding_idx(self):
        if self.padding_label is None:
            return None
        return self.word2idx[self.padding_label]

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

