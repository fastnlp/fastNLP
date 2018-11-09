import torch
import numpy as np


class Field(object):
    """A field defines a data type.

    """

    def __init__(self, content, is_target: bool):
        self.is_target = is_target
        self.content = content

    def index(self, vocab):
        """create index field
        """
        raise NotImplementedError

    def __len__(self):
        """number of samples
        """
        assert self.content is not None
        return len(self.content)

    def to_tensor(self, id_list):
        """convert batch of index to tensor
        """
        raise NotImplementedError

    def __repr__(self):
        return self.content.__repr__()

class TextField(Field):
    def __init__(self, text, is_target):
        """
        :param text: list of strings
        :param is_target: bool
        """
        super(TextField, self).__init__(text, is_target)


class LabelField(Field):
    """The Field representing a single label. Can be a string or integer.

    """
    def __init__(self, label, is_target=True):
        super(LabelField, self).__init__(label, is_target)


class SeqLabelField(Field):
    def __init__(self, label_seq, is_target=True):
        super(SeqLabelField, self).__init__(label_seq, is_target)


class CharTextField(Field):
    def __init__(self, text, max_word_len, is_target=False):
        super(CharTextField, self).__init__(is_target)
        # TODO
        raise NotImplementedError
        self.max_word_len = max_word_len
        self._index = []

    def get_length(self):
        return len(self.text)

    def contents(self):
        return self.text.copy()

    def index(self, char_vocab):
        if len(self._index) == 0:
            for word in self.text:
                char_index = [char_vocab[ch] for ch in word]
                if self.max_word_len >= len(char_index):
                    char_index += [0] * (self.max_word_len - len(char_index))
                else:
                    self._index.clear()
                    raise RuntimeError("Word {} has more than {} characters. ".format(word, self.max_word_len))
                self._index.append(char_index)
        return self._index

    def to_tensor(self, padding_length):
        """

        :param padding_length: int, the padding length of the word sequence.
        :return : tensor of shape (padding_length, max_word_len)
        """
        pads = [[0] * self.max_word_len] * (padding_length - self.get_length())
        return torch.LongTensor(self._index + pads)
