from copy import deepcopy

DEFAULT_PADDING_LABEL = '<pad>'  # dict index = 0
DEFAULT_UNKNOWN_LABEL = '<unk>'  # dict index = 1
DEFAULT_RESERVED_LABEL = ['<reserved-2>',
                          '<reserved-3>',
                          '<reserved-4>']  # dict index = 2~4

DEFAULT_WORD_TO_INDEX = {DEFAULT_PADDING_LABEL: 0, DEFAULT_UNKNOWN_LABEL: 1,
                         DEFAULT_RESERVED_LABEL[0]: 2, DEFAULT_RESERVED_LABEL[1]: 3,
                         DEFAULT_RESERVED_LABEL[2]: 4}


def isiterable(p_object):
    try:
        it = iter(p_object)
    except TypeError:
        return False
    return True

def check_build_vocab(func):
    def _wrapper(self, *args, **kwargs):
        if self.word2idx is None:
            self.build_vocab()
            self.build_reverse_vocab()
        elif self.idx2word is None:
            self.build_reverse_vocab()
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
        self.word_count = {}
        self.has_default = need_default
        self.word2idx = None
        self.idx2word = None


    def update(self, word):
        """add word or list of words into Vocabulary

        :param word: a list of string or a single string
        """
        if not isinstance(word, str) and isiterable(word):
            # it's a nested list
            for w in word:
                self.update(w)
        else:
        # it's a word to be added
            if word not in self.word_count:
                self.word_count[word] = 1
            else:
                self.word_count[word] += 1
            self.word2idx = None


    def build_vocab(self):
        """build 'word to index' dict, and filter the word using `max_size` and `min_freq`
        """
        if self.has_default:
            self.word2idx = deepcopy(DEFAULT_WORD_TO_INDEX)
            self.padding_label = DEFAULT_PADDING_LABEL
            self.unknown_label = DEFAULT_UNKNOWN_LABEL
        else:
            self.word2idx = {}
            self.padding_label = None
            self.unknown_label = None

        words = sorted(self.word_count.items(), key=lambda kv: kv[1], reverse=True)
        if self.min_freq is not None:
            words = list(filter(lambda kv: kv[1] >= self.min_freq, words))
        if self.max_size is not None and len(words) > self.max_size:
            words = words[:self.max_size]
        for w, _ in words:
            self.word2idx[w] = len(self.word2idx)

    def build_reverse_vocab(self):
        """build 'index to word' dict based on 'word to index' dict
        """
        self.idx2word = {self.word2idx[w] : w for w in self.word2idx}

    @check_build_vocab
    def __len__(self):
        return len(self.word2idx)

    @check_build_vocab
    def has_word(self, w):
        return w in self.word2idx

    @check_build_vocab
    def __getitem__(self, w):
        """To support usage like::

            vocab[w]
        """
        if w in self.word2idx:
            return self.word2idx[w]
        elif self.has_default:
            return self.word2idx[DEFAULT_UNKNOWN_LABEL]
        else:
            raise ValueError("word {} not in vocabulary".format(w))

    @check_build_vocab
    def to_index(self, w):
        """ like to_index(w) function, turn a word to the index
            if w is not in Vocabulary, return the unknown label

        :param str w:
        """
        return self[w]

    @property
    @check_build_vocab
    def unknown_idx(self):
        if self.unknown_label is None:
            return None
        return self.word2idx[self.unknown_label]

    @property
    @check_build_vocab
    def padding_idx(self):
        if self.padding_label is None:
            return None
        return self.word2idx[self.padding_label]

    @check_build_vocab
    def to_word(self, idx):
        """given a word's index, return the word itself

        :param int idx:
        """
        if self.idx2word is None:
            self.build_reverse_vocab()
        return self.idx2word[idx]

    def __getstate__(self):
        """use to prepare data for pickle
        """
        state = self.__dict__.copy()
        # no need to pickle idx2word as it can be constructed from word2idx
        del state['idx2word']
        return state

    def __setstate__(self, state):
        """use to restore state from pickle
        """
        self.__dict__.update(state)
        self.idx2word = None
