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

class Vocabulary(object):
    def __init__(self):
        self.word2idx = deepcopy(DEFAULT_WORD_TO_INDEX)
        self.padding_label = DEFAULT_PADDING_LABEL
        self.unknown_label = DEFAULT_UNKNOWN_LABEL
        self.idx2word = None

    def __len__(self):
        return len(self.word2idx)
    
    def update(self, word):
        """add word or list of words into Vocabulary
        """
        if not isinstance(word, str) and isiterable(word):
        # it's a nested list
            for w in word:
                self.update(w)
        else:
        # it's a word to be added
            self.word2idx[word] = len(self)
            if self.idx2word is not None:
                self.idx2word = None

    
    def __getitem__(self, w):
        """ like to_index(w) function, turn a word to the index
            if w is not in Vocabulary, return the unknown label
        """
        if w in self.word2idx:
            return self.word2idx[w]
        else:
            return self.word2idx[DEFAULT_UNKNOWN_LABEL]
    
    def unknown_idx(self):
        return self.word2idx[self.unknown_label]
    
    def padding_idx(self):
        return self.word2idx[self.padding_label]

    def build_reverse_vocab(self):
        self.idx2word = {self.word2idx[w] : w for w in self.word2idx}
    
    def to_word(self, idx):
        """given a word's index, return the word itself
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
    
if __name__ == '__main__':
    import _pickle as pickle
    vocab = Vocabulary()
    filename = 'vocab'
    vocab.update(filename)
    vocab.update([filename, ['a'], [['b']], ['c']])
    idx = vocab[filename]
    print('{} {}'.format(vocab.to_word(idx), vocab[filename]))

    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
    
    print('{} {}'.format(vocab.to_word(idx), vocab[filename]))
    print(vocab.word2idx)
    