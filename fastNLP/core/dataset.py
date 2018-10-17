import random
import sys
from collections import defaultdict
from copy import deepcopy

from fastNLP.core.field import TextField, LabelField
from fastNLP.core.instance import Instance
from fastNLP.core.vocabulary import Vocabulary

class DataSet(list):
    """A DataSet object is a list of Instance objects.

    """

    def __init__(self, name="", instances=None):
        """

        :param name: str, the name of the dataset. (default: "")
        :param instances: list of Instance objects. (default: None)
        """
        list.__init__([])
        self.name = name
        self.origin_len = None
        if instances is not None:
            self.extend(instances)

    def index_all(self, vocab):
        for ins in self:
            ins.index_all(vocab)
        return self

    def index_field(self, field_name, vocab):
        for ins in self:
            ins.index_field(field_name, vocab)
        return self

    def to_tensor(self, idx: int, padding_length: dict):
        """Convert an instance in a dataset to tensor.

        :param idx: int, the index of the instance in the dataset.
        :param padding_length: int
        :return tensor_x: dict of (str: torch.LongTensor), which means (field name: tensor of shape [padding_length, ])
                tensor_y: dict of (str: torch.LongTensor), which means (field name: tensor of shape [padding_length, ])

        """
        ins = self[idx]
        return ins.to_tensor(padding_length, self.origin_len)

    def get_length(self):
        """Fetch lengths of all fields in all instances in a dataset.

        :return lengths: dict of (str: list). The str is the field name.
                The list contains lengths of this field in all instances.

        """
        lengths = defaultdict(list)
        for ins in self:
            for field_name, field_length in ins.get_length().items():
                lengths[field_name].append(field_length)
        return lengths

    def shuffle(self):
        random.shuffle(self)
        return self

    def split(self, ratio, shuffle=True):
        """Train/dev splitting

        :param ratio: float, between 0 and 1. The ratio of development set in origin data set.
        :param shuffle: bool, whether shuffle the data set before splitting. Default: True.
        :return train_set: a DataSet object, representing the training set
                dev_set: a DataSet object, representing the validation set

        """
        assert 0 < ratio < 1
        if shuffle:
            self.shuffle()
        split_idx = int(len(self) * ratio)
        dev_set = deepcopy(self)
        train_set = deepcopy(self)
        del train_set[:split_idx]
        del dev_set[split_idx:]
        return train_set, dev_set

    def rename_field(self, old_name, new_name):
        """rename a field
        """
        for ins in self:
            ins.rename_field(old_name, new_name)
        return self

    def set_target(self, **fields):
        """Change the flag of `is_target` for all instance. For fields not set here, leave their `is_target` unchanged.

        :param key-value pairs for field-name and `is_target` value(True, False or None).
        """
        for ins in self:
            ins.set_target(**fields)
        return self

    def update_vocab(self, **name_vocab):
        for field_name, vocab in name_vocab.items():
            for ins in self:
                vocab.update(ins[field_name].contents())
        return self

    def set_origin_len(self, origin_field, origin_len_name=None):
        if origin_field is None:
            self.origin_len = None
        else:
            self.origin_len = (origin_field + "_origin_len", origin_field) \
                if origin_len_name is None else (origin_len_name, origin_field)
        return self
