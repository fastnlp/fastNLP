from collections import defaultdict


class DataSet(list):
    def __init__(self, name="", instances=None):
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
