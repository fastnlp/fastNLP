import torch
import numpy as np


class Field(object):
    """A field defines a data type.

    """

    def __init__(self, name, is_target: bool):
        self.name = name
        self.is_target = is_target
        self.content = None

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

class TextField(Field):
    def __init__(self, name, text, is_target):
        """
        :param text: list of strings
        :param is_target: bool
        """
        super(TextField, self).__init__(name, is_target)
        self.content = text

    def index(self, vocab):
        idx_field = IndexField(self.name+'_idx', self.content, vocab, self.is_target)
        return idx_field


class IndexField(Field):
    def __init__(self, name, content, vocab, is_target):
        super(IndexField, self).__init__(name, is_target)
        self.content = []
        self.padding_idx = vocab.padding_idx
        for sent in content:
            idx = vocab.index_sent(sent)
            if isinstance(idx, list):
                idx = torch.Tensor(idx)
            elif isinstance(idx, np.array):
                idx = torch.from_numpy(idx)
            elif not isinstance(idx, torch.Tensor):
                raise ValueError
            self.content.append(idx)

    def to_tensor(self, id_list, sort_within_batch=False):
        max_len = max(id_list)
        batch_size = len(id_list)
        tensor = torch.full((batch_size, max_len), self.padding_idx, dtype=torch.long)
        len_list = [(i, self.content[i].size(0)) for i in id_list]
        if sort_within_batch:
            len_list = sorted(len_list, key=lambda x: x[1], reverse=True)
        for i, (idx, length) in enumerate(len_list):
            if length == max_len:
                tensor[i] = self.content[idx]
            else:
                tensor[i][:length] = self.content[idx]
        return tensor

class LabelField(Field):
    """The Field representing a single label. Can be a string or integer.

    """
    def __init__(self, label, is_target=True):
        super(LabelField, self).__init__(is_target)
        self.label = label
        self._index = None

    def get_length(self):
        """Fetch the length of the label field.

        :return length: int, the length of the label, always 1.
        """
        return 1

    def index(self, vocab):
        if self._index is None:
            if isinstance(self.label, str):
                self._index = vocab[self.label]
        return self._index

    def to_tensor(self, padding_length):
        if self._index is None:
            if isinstance(self.label, int):
                return torch.tensor(self.label)
            elif isinstance(self.label, str):
                raise RuntimeError("Field {} not indexed. Call index method.".format(self.label))
            else:
                raise RuntimeError(
                    "Not support type for LabelField. Expect str or int, got {}.".format(type(self.label)))
        else:
            return torch.LongTensor([self._index])

    def contents(self):
        return [self.label]

class SeqLabelField(Field):
    def __init__(self, label_seq, is_target=True):
        super(SeqLabelField, self).__init__(is_target)
        self.label_seq = label_seq
        self._index = None

    def get_length(self):
        return len(self.label_seq)

    def index(self, vocab):
        if self._index is None:
            self._index = [vocab[c] for c in self.label_seq]
        return self._index

    def to_tensor(self, padding_length):
        pads = [0] * (padding_length - self.get_length())
        if self._index is None:
            if self.get_length() == 0:
                return torch.LongTensor(pads)
            elif isinstance(self.label_seq[0], int):
                return torch.LongTensor(self.label_seq + pads)
            elif isinstance(self.label_seq[0], str):
                raise RuntimeError("Field {} not indexed. Call index method.".format(self.label))
            else:
                raise RuntimeError(
                    "Not support type for SeqLabelField. Expect str or int, got {}.".format(type(self.label)))
        else:
            return torch.LongTensor(self._index + pads)

    def contents(self):
        return self.label_seq.copy()


class CharTextField(Field):
    def __init__(self, text, max_word_len, is_target=False):
        super(CharTextField, self).__init__(is_target)
        self.text = text
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
