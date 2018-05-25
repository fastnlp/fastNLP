import numpy as np


class BaseModel(object):
    """PyTorch base model for all models"""

    def __init__(self):
        pass

    def prepare_input(self, data):
        """
        :param data: str, raw input vector(?)
        :return (X, Y): tuple, input features and labels
        """
        raise NotImplementedError

    def mode(self, test=False):
        raise NotImplementedError

    def data_forward(self, *x):
        # required by PyTorch nn
        raise NotImplementedError

    def grad_backward(self):
        raise NotImplementedError

    def get_loss(self, pred, truth):
        raise NotImplementedError


class ToyModel(BaseModel):
    """This is for code testing."""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.test_mode = False
        self.weight = np.random.rand(5, 1)
        self.bias = np.random.rand()
        self._loss = 0

    def prepare_input(self, data):
        return data[:, :-1], data[:, -1]

    def mode(self, test=False):
        self.test_mode = test

    def data_forward(self, x):
        return np.matmul(x, self.weight) + self.bias

    def grad_backward(self):
        print("loss gradient backward")

    def get_loss(self, pred, truth):
        self._loss = np.mean(np.square(pred - truth))
        return self._loss


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
