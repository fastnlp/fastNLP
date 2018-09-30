import random
from collections import defaultdict
from copy import deepcopy

from fastNLP.core.field import TextField, LabelField
from fastNLP.core.instance import Instance
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.loader.dataset_loader import POSDataSetLoader, ClassDataSetLoader


def create_dataset_from_lists(str_lists: list, word_vocab: dict, has_target: bool = False, label_vocab: dict = None):
    if has_target is True:
        if label_vocab is None:
            raise RuntimeError("Must provide label vocabulary to transform labels.")
        return create_labeled_dataset_from_lists(str_lists, word_vocab, label_vocab)
    else:
        return create_unlabeled_dataset_from_lists(str_lists, word_vocab)


def create_labeled_dataset_from_lists(str_lists, word_vocab, label_vocab):
    """Create an DataSet instance that contains labels.

    :param str_lists: list of list of strings, [num_examples, 2, *].
            ::
            [
                [[word_11, word_12, ...], [label_11, label_12, ...]],
                ...
            ]

    :param word_vocab: dict of (str: int), which means (word: index).
    :param label_vocab: dict of (str: int), which means (word: index).
    :return data_set: a DataSet instance.

    """
    data_set = DataSet()
    for example in str_lists:
        word_seq, label_seq = example[0], example[1]
        x = TextField(word_seq, is_target=False)
        y = TextField(label_seq, is_target=True)
        data_set.append(Instance(word_seq=x, label_seq=y))
    data_set.index_field("word_seq", word_vocab)
    data_set.index_field("label_seq", label_vocab)
    return data_set


def create_unlabeled_dataset_from_lists(str_lists, word_vocab):
    """Create an DataSet instance that contains no labels.

    :param str_lists: list of list of strings, [num_examples, *].
            ::
            [
                [word_11, word_12, ...],
                ...
            ]

    :param word_vocab: dict of (str: int), which means (word: index).
    :return data_set: a DataSet instance.

    """
    data_set = DataSet()
    for word_seq in str_lists:
        x = TextField(word_seq, is_target=False)
        data_set.append(Instance(word_seq=x))
    data_set.index_field("word_seq", word_vocab)
    return data_set


class DataSet(list):
    """A DataSet object is a list of Instance objects.

    """

    def __init__(self, name="", instances=None, loader=None):
        """

        :param name: str, the name of the dataset. (default: "")
        :param instances: list of Instance objects. (default: None)

        """
        list.__init__([])
        self.name = name
        if instances is not None:
            self.extend(instances)
        self.dataset_loader = loader

    def index_all(self, vocab):
        for ins in self:
            ins.index_all(vocab)

    def index_field(self, field_name, vocab):
        for ins in self:
            ins.index_field(field_name, vocab)

    def to_tensor(self, idx: int, padding_length: dict):
        """Convert an instance in a dataset to tensor.

        :param idx: int, the index of the instance in the dataset.
        :param padding_length: int
        :return tensor_x: dict of (str: torch.LongTensor), which means (field name: tensor of shape [padding_length, ])
                tensor_y: dict of (str: torch.LongTensor), which means (field name: tensor of shape [padding_length, ])

        """
        ins = self[idx]
        return ins.to_tensor(padding_length)

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

    def convert(self, data):
        """Convert lists of strings into Instances with Fields"""
        raise NotImplementedError

    def convert_with_vocabs(self, data, vocabs):
        """Convert lists of strings into Instances with Fields, using existing Vocabulary. Useful in predicting."""
        raise NotImplementedError

    def convert_for_infer(self, data, vocabs):
        """Convert lists of strings into Instances with Fields."""

    def load(self, data_path, vocabs=None, infer=False):
        """Load data from the given files.

        :param data_path: str, the path to the data
        :param infer: bool. If True, there is no label information in the data. Default: False.
        :param vocabs: dict of (name: Vocabulary object), used to index data. If not provided, a new vocabulary will be constructed.

        """
        raw_data = self.dataset_loader.load(data_path)
        if infer is True:
            self.convert_for_infer(raw_data, vocabs)
        else:
            if vocabs is not None:
                self.convert_with_vocabs(raw_data, vocabs)
            else:
                self.convert(raw_data)

    def load_raw(self, raw_data, vocabs):
        """

        :param raw_data:
        :param vocabs:
        :return:
        """
        self.convert_for_infer(raw_data, vocabs)

    def split(self, ratio, shuffle=True):
        """Train/dev splitting

        :param ratio: float, between 0 and 1. The ratio of development set in origin data set.
        :param shuffle: bool, whether shuffle the data set before splitting. Default: True.
        :return train_set: a DataSet object, representing the training set
                dev_set: a DataSet object, representing the validation set

        """
        assert 0 < ratio < 1
        if shuffle:
            random.shuffle(self)
        split_idx = int(len(self) * ratio)
        dev_set = deepcopy(self)
        train_set = deepcopy(self)
        del train_set[:split_idx]
        del dev_set[split_idx:]
        return train_set, dev_set


class SeqLabelDataSet(DataSet):
    def __init__(self, instances=None, loader=POSDataSetLoader()):
        super(SeqLabelDataSet, self).__init__(name="", instances=instances, loader=loader)
        self.word_vocab = Vocabulary()
        self.label_vocab = Vocabulary()

    def convert(self, data):
        """Convert lists of strings into Instances with Fields.

        :param data: 3-level lists. Entries are strings.
        """
        for example in data:
            word_seq, label_seq = example[0], example[1]
            # list, list
            self.word_vocab.update(word_seq)
            self.label_vocab.update(label_seq)
            x = TextField(word_seq, is_target=False)
            x_len = LabelField(len(word_seq), is_target=False)
            y = TextField(label_seq, is_target=False)
            instance = Instance()
            instance.add_field("word_seq", x)
            instance.add_field("truth", y)
            instance.add_field("word_seq_origin_len", x_len)
            self.append(instance)
        self.index_field("word_seq", self.word_vocab)
        self.index_field("truth", self.label_vocab)
        # no need to index "word_seq_origin_len"

    def convert_with_vocabs(self, data, vocabs):
        for example in data:
            word_seq, label_seq = example[0], example[1]
            # list, list
            x = TextField(word_seq, is_target=False)
            x_len = LabelField(len(word_seq), is_target=False)
            y = TextField(label_seq, is_target=False)
            instance = Instance()
            instance.add_field("word_seq", x)
            instance.add_field("truth", y)
            instance.add_field("word_seq_origin_len", x_len)
            self.append(instance)
        self.index_field("word_seq", vocabs["word_vocab"])
        self.index_field("truth", vocabs["label_vocab"])
        # no need to index "word_seq_origin_len"

    def convert_for_infer(self, data, vocabs):
        for word_seq in data:
            # list
            x = TextField(word_seq, is_target=False)
            x_len = LabelField(len(word_seq), is_target=False)
            instance = Instance()
            instance.add_field("word_seq", x)
            instance.add_field("word_seq_origin_len", x_len)
            self.append(instance)
        self.index_field("word_seq", vocabs["word_vocab"])
        # no need to index "word_seq_origin_len"


class TextClassifyDataSet(DataSet):
    def __init__(self, instances=None, loader=ClassDataSetLoader()):
        super(TextClassifyDataSet, self).__init__(name="", instances=instances, loader=loader)
        self.word_vocab = Vocabulary()
        self.label_vocab = Vocabulary(need_default=False)

    def convert(self, data):
        for example in data:
            word_seq, label = example[0], example[1]
            # list, str
            self.word_vocab.update(word_seq)
            self.label_vocab.update(label)
            x = TextField(word_seq, is_target=False)
            y = LabelField(label, is_target=True)
            instance = Instance()
            instance.add_field("word_seq", x)
            instance.add_field("label", y)
            self.append(instance)
        self.index_field("word_seq", self.word_vocab)
        self.index_field("label", self.label_vocab)

    def convert_with_vocabs(self, data, vocabs):
        for example in data:
            word_seq, label = example[0], example[1]
            # list, str
            x = TextField(word_seq, is_target=False)
            y = LabelField(label, is_target=True)
            instance = Instance()
            instance.add_field("word_seq", x)
            instance.add_field("label", y)
            self.append(instance)
        self.index_field("word_seq", vocabs["word_vocab"])
        self.index_field("label", vocabs["label_vocab"])

    def convert_for_infer(self, data, vocabs):
        for word_seq in data:
            # list
            x = TextField(word_seq, is_target=False)
            instance = Instance()
            instance.add_field("word_seq", x)
            self.append(instance)
        self.index_field("word_seq", vocabs["word_vocab"])


def change_field_is_target(data_set, field_name, new_target):
    """Change the flag of is_target in a field.

    :param data_set: a DataSet object
    :param field_name: str, the name of the field
    :param new_target: one of (True, False, None), representing this field is batch_x / is batch_y / neither.

    """
    for inst in data_set:
        inst.fields[field_name].is_target = new_target


if __name__ == "__main__":
    data_set = SeqLabelDataSet()
    data_set.load("../../test/data_for_tests/people.txt")
    a, b = data_set.split(0.3)
    print(type(data_set), type(a), type(b))
    print(len(data_set), len(a), len(b))
