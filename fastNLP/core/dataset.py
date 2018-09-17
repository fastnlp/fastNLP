from collections import defaultdict

from fastNLP.core.field import TextField
from fastNLP.core.instance import Instance


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
    def __init__(self, name="", instances=None):
        """

        :param name: str, the name of the dataset. (default: "")
        :param instances: list of Instance objects. (default: None)

        """
        list.__init__([])
        self.name = name
        if instances is not None:
            self.extend(instances)

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
