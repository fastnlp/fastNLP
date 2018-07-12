import collections
import pickle

class Word2Idx():
    """
    Build a word index according to word frequency.

    If "min_freq" is given, then only words with a frequncy not lesser than min_freq will be kept.
    If "max_num" is given, then at most the most frequent $max_num words will be kept.
    "words" should be a list [ w_1,w_2,...,w_i,...,w_n ] where each w_i is a string representing a word.
    num is the size of the lookup table.
    w2i is a lookup table assigning each word an index.
    i2w is a vector which serves as an invert mapping of w2i.
    Note that index 0 is token "<PAD>" for padding
    index 1 is token "<UNK>" for unregistered words
    e.g. i2w[w2i["word"]] == "word"
    """
    def __init__(self):
        self.__w2i = dict()
        self.__i2w = []
        self.num = 0

    def build(self, words, min_freq=0, max_num=None):
        """build a model from words"""
        counter = collections.Counter(words)
        word_set = set(words)
        if max_num is not None:
            most_common = counter.most_common(min(len(word_set), max_num - 1))
        else:
            most_common = counter.most_common()
        self.__w2i = dict((w[0],i + 1) for i,w in enumerate(most_common) if w[1] >= min_freq)
        self.__w2i["<PAD>"] = 0
        self.__w2i["<UNK>"] = 1
        self.__i2w = ["<PAD>", "<UNK>"] + [ w[0] for w in most_common if w[1] >= min_freq ]
        self.num = len(self.__i2w)

    def w2i(self, word):
        """word to index"""
        if word in self.__w2i:
            return self.__w2i[word]
        return 0

    def i2w(self, idx):
        """index to word"""
        if idx >= self.num:
            raise Exception("out of range\n")
        return self.__i2w[idx]

    def save(self, addr):
        """save the model to a file with address "addr" """
        f = open(addr,"wb")
        pickle.dump([self.__i2w, self.__w2i, self.num], f)
        f.close()

    def load(self, addr):
        """load a model from a file with address "addr" """
        f = open(addr,"rb")
        paras = pickle.load(f)
        self.__i2w, self.__w2i, self.num = paras[0], paras[1], paras[2]
        f.close()

    

