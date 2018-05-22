class BaseModel(object):
    """base model for all models"""

    def __init__(self):
        pass

    def prepare_input(self, data):
        raise NotImplementedError

    def mode(self, test=False):
        raise NotImplementedError

    def data_forward(self, x):
        raise NotImplementedError

    def grad_backward(self):
        raise NotImplementedError

    def loss(self, pred, truth):
        raise NotImplementedError


class Vocabulary(object):
    """
        A collection of lookup tables.
    """

    def __init__(self):
        self.word_set = None
        self.word2idx = None
        self.emb_matrix = None

    def lookup(self, word):
        if word in self.word_set:
            return self.emb_matrix[self.word2idx[word]]
        return LookupError("The key " + word + " does not exist.")


class Document(object):
    """
        contains a sequence of tokens
        each token is a character with linguistic attributes
    """

    def __init__(self):
        # wrap pandas.dataframe
        self.dataframe = None
