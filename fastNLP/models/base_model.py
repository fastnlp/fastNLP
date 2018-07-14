import torch


class BaseModel(torch.nn.Module):
    """Base PyTorch model for all models.
        To do: add some useful common features
    """

    def __init__(self):
        super(BaseModel, self).__init__()


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

