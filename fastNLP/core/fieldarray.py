import numpy as np


class FieldArray(object):
    """FieldArray is the collection of Instances of the same Field.
    It is the basic element of DataSet class.

    """
    def __init__(self, name, content, padding_val=0, is_target=False, is_input=False):
        """

        :param str name: the name of the FieldArray
        :param list content: a list of int, float, or other objects.
        :param int padding_val: the integer for padding. Default: 0.
        :param bool is_target: If True, this FieldArray is used to compute loss.
        :param bool is_input: If True, this FieldArray is used to the model input.
        """
        self.name = name
        self.content = content
        self.padding_val = padding_val
        self.is_target = is_target
        self.is_input = is_input
        self.dtype = None

    def __repr__(self):
        return "FieldArray {}: {}".format(self.name, self.content.__repr__())

    def append(self, val):
        self.content.append(val)

    def __getitem__(self, name):
        return self.get(name)

    def __setitem__(self, name, val):
        assert isinstance(name, int)
        self.content[name] = val

    def get(self, indices):
        """Fetch instances based on indices.

        :param indices: an int, or a list of int.
        :return:
        """
        if isinstance(indices, int):
            return self.content[indices]
        assert self.is_input is True or self.is_target is True
        batch_size = len(indices)
        # TODO 当这个fieldArray是seq_length这种只有一位的内容时，不需要padding，需要再讨论一下
        if not isiterable(self.content[0]):
            if self.dtype is None:
                self.dtype = np.int64 if isinstance(self.content[0], int) else np.double
            array = np.array([self.content[i] for i in indices], dtype=self.dtype)
        else:
            if self.dtype is None:
                self.dtype = np.int64
            max_len = max([len(self.content[i]) for i in indices])
            array = np.full((batch_size, max_len), self.padding_val, dtype=self.dtype)

            for i, idx in enumerate(indices):
                array[i][:len(self.content[idx])] = self.content[idx]
        return array

    def __len__(self):
        return len(self.content)

def isiterable(content):
    try:
        _ = (e for e in content)
    except TypeError:
        return False
    return True