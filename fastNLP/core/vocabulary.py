from collections import Counter
from fastNLP.core.dataset import DataSet

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

    :param int max_size: set the max number of words in Vocabulary. Default: None
    :param int min_freq: set the min occur frequency of words in Vocabulary. Default: None
    :param padding: str, padding的字符，默认为<pad>。如果设置为None，则vocabulary中不考虑padding，为None的情况多在为label建立
        Vocabulary的情况。
    :param unknown: str, unknown的字符，默认为<unk>。如果设置为None，则vocabulary中不考虑unknown，为None的情况多在为label建立
        Vocabulary的情况。

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
        """Build a mapping from word to index, and filter the word using ``max_size`` and ``min_freq``.

        """
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
        """Build "index to word" dict based on "word to index" dict.

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

    @check_build_vocab
    def index_dataset(self, *datasets, field_name, new_field_name=None):
        """
        example:
            # remember to use `field_name`
            vocab.index_dataset(tr_data, dev_data, te_data, field_name='words')

        :param datasets: fastNLP Dataset type. you can pass multiple datasets
        :param field_name: str, what field to index. Only support 0,1,2 dimension.
        :param new_field_name: str. What the indexed field should be named, default is to overwrite field_name
        :return:
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
                    return[[self.to_index(c) for c in w] for w in field]

        if new_field_name is None:
            new_field_name = field_name
        for dataset in datasets:
            if isinstance(dataset, DataSet):
                dataset.apply(index_instance, new_field_name=new_field_name)
            else:
                raise RuntimeError("Only DataSet type is allowed.")

    def from_dataset(self, *datasets, field_name):
        """
        Construct vocab from dataset.

        :param datasets: DataSet.
        :param field_name: str, what field is used to construct dataset.
        :return:
        """
        def construct_vocab(ins):
            field = ins[field_name]
            if isinstance(field, str):
                self.add_word(field)
            elif isinstance(field, list):
                if not isinstance(field[0], list):
                    self.add_word_lst(field)
                else:
                    if isinstance(field[0][0], list):
                        raise RuntimeError("Only support field with 2 dimensions.")
                    [self.add_word_lst(w) for w in field]
        for dataset in datasets:
            if isinstance(dataset, DataSet):
                dataset.apply(construct_vocab)
            else:
                raise RuntimeError("Only DataSet type is allowed.")

    def to_index(self, w):
        """ Turn a word to an index. If w is not in Vocabulary, return the unknown label.

        :param str w: a word
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

    def __repr__(self):
        return "Vocabulary({}...)".format(list(self.word_count.keys())[:5])

    def __iter__(self):
        return iter(list(self.word_count.keys()))
