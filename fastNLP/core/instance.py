class Instance(object):
    """An instance which consists of Fields is an example in the DataSet.

    """

    def __init__(self, **fields):
        self.fields = fields
        self.has_index = False
        self.indexes = {}

    def add_field(self, field_name, field):
        self.fields[field_name] = field

    def get_length(self):
        """Fetch the length of all fields in the instance.

        :return length: dict of (str: int), which means (field name: field length).

        """
        length = {name: field.get_length() for name, field in self.fields.items()}
        return length

    def index_field(self, field_name, vocab):
        """use `vocab` to index certain field
        """
        self.indexes[field_name] = self.fields[field_name].index(vocab)

    def index_all(self, vocab):
        """use `vocab` to index all fields
        """
        if self.has_index:
            print("error")
            return self.indexes
        indexes = {name: field.index(vocab) for name, field in self.fields.items()}
        self.indexes = indexes
        return indexes

    def to_tensor(self, padding_length: dict):
        """Convert instance to tensor.

        :param padding_length: dict of (str: int), which means (field name: padding_length of this field)
        :return tensor_x: dict of (str: torch.LongTensor), which means (field name: tensor of shape [padding_length, ])
                tensor_y: dict of (str: torch.LongTensor), which means (field name: tensor of shape [padding_length, ])
                        If is_target is False for all fields, tensor_y would be an empty dict.
        """
        tensor_x = {}
        tensor_y = {}
        for name, field in self.fields.items():
            if field.is_target:
                tensor_y[name] = field.to_tensor(padding_length[name])
            else:
                tensor_x[name] = field.to_tensor(padding_length[name])
        return tensor_x, tensor_y
