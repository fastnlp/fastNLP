import numpy as np
import torch


class BaseModel(torch.nn.Module):
    """Base PyTorch model for all models.
        Three network modules presented:
            - embedding module
            - aggregation module
            - output module
        Subclasses must implement these three modules with "components".
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *inputs):
        x = self.encode(*inputs)
        x = self.aggregation(x)
        x = self.output(x)
        return x

    def encode(self, x):
        raise NotImplementedError

    def aggregation(self, x):
        raise NotImplementedError

    def output(self, x):
        raise NotImplementedError


class BaseController(object):
    """Base Controller for all controllers.
        This class and its subclasses are actually "controllers" of the PyTorch models.
        They act as an interface between Trainer and the PyTorch models.
        This controller provides the following methods to be called by Trainer.
        - prepare_input
        - mode
        - define_optimizer
        - data_forward
        - grad_backward
        - get_loss
    """

    def __init__(self):
        """
        Define PyTorch model parameters here.
        """
        pass

    def prepare_input(self, data):
        """
        Perform data transformation from raw input to vector/matrix inputs.
        :param data: raw inputs
        :return (X, Y): tuple, input features and labels
        """
        raise NotImplementedError

    def mode(self, test=False):
        """
        Tell the network to be trained or not, required by PyTorch.
        :param test: bool
        """
        raise NotImplementedError

    def define_optimizer(self):
        """
        Define PyTorch optimizer specified by the models.
        """
        raise NotImplementedError

    def data_forward(self, *x):
        """
        Forward pass of the data.
        :param x: input feature matrix and label vector
        :return: output by the models
        """
        # required by PyTorch nn
        raise NotImplementedError

    def grad_backward(self):
        """
        Perform gradient descent to update the models parameters.
        """
        raise NotImplementedError

    def get_loss(self, pred, truth):
        """
        Compute loss given models prediction and ground truth. Loss function specified by the models.
        :param pred: prediction label vector
        :param truth: ground truth label vector
        :return: a scalar
        """
        raise NotImplementedError


class ToyController(BaseController):
    """This is for code testing."""

    def __init__(self):
        super(ToyController, self).__init__()
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

    def define_optimizer(self):
        pass


class Vocabulary(object):
    """A look-up table that allows you to access `Lexeme` objects. The `Vocab`
    instance also provides access to the `StringStore`, and owns underlying
    data that is shared between `Doc` objects.
    """

    def __init__(self):
        """Create the vocabulary.
        RETURNS (Vocab): The newly constructed object.
        """
        self.data_frame = None


class Document(object):
    """A sequence of Token objects. Access sentences and named entities, export
    annotations to numpy arrays, losslessly serialize to compressed binary
    strings. The `Doc` object holds an array of `Token` objects. The
    Python-level `Token` and `Span` objects are views of this array, i.e.
    they don't own the data themselves. -- spacy
    """

    def __init__(self, vocab, words=None, spaces=None):
        """Create a Doc object.
        vocab (Vocab): A vocabulary object, which must match any models you
            want to use (e.g. tokenizer, parser, entity recognizer).
        words (list or None): A list of unicode strings, to add to the document
            as words. If `None`, defaults to empty list.
        spaces (list or None): A list of boolean values, of the same length as
            words. True means that the word is followed by a space, False means
            it is not. If `None`, defaults to `[True]*len(words)`
        user_data (dict or None): Optional extra data to attach to the Doc.
        RETURNS (Doc): The newly constructed object.
        """
        self.vocab = vocab
        self.spaces = spaces
        self.words = words
        if spaces is None:
            self.spaces = [True] * len(self.words)
        elif len(spaces) != len(self.words):
            raise ValueError("dismatch spaces and words")

    def get_chunker(self, vocab):
        return None

    def push_back(self, vocab):
        pass


class Token(object):
    """An individual token – i.e. a word, punctuation symbol, whitespace,
    etc.
    """

    def __init__(self, vocab, doc, offset):
        """Construct a `Token` object.
            vocab (Vocabulary): A storage container for lexical types.
            doc (Document): The parent document.
            offset (int): The index of the token within the document.
        """
        self.vocab = vocab
        self.doc = doc
        self.token = doc[offset]
        self.i = offset
