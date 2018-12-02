import numpy as np


class FieldArray(object):
    """FieldArray is the collection of Instances of the same Field.
    It is the basic element of DataSet class.

    """

    def __init__(self, name, content, padding_val=0, is_target=False, is_input=False):
        """

        :param str name: the name of the FieldArray
        :param list content: a list of int, float, or a list of list.
        :param int padding_val: the integer for padding. Default: 0.
        :param bool is_target: If True, this FieldArray is used to compute loss.
        :param bool is_input: If True, this FieldArray is used to the model input.
        """
        self.name = name
        self.content = content
        self.padding_val = padding_val
        self.is_target = is_target
        self.is_input = is_input
        self.pytype = self._type_detection(content)
        self.dtype = self._map_to_np_type(self.pytype)

    @staticmethod
    def _type_detection(content):

        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], list):
            # 2-D list
            # TODO: refactor
            type_set = set([type(item) for item in content[0]])
        else:
            # 1-D list
            if len(content) == 0:
                raise RuntimeError("Cannot create FieldArray with an empty list.")
            type_set = set([type(item) for item in content])

        if len(type_set) == 1 and any(basic_type in type_set for basic_type in (str, int, float)):
            return type_set.pop()
        elif len(type_set) == 2 and float in type_set and int in type_set:
            # up-cast int to float
            for idx, _ in enumerate(content):
                content[idx] = float(content[idx])
            return float
        else:
            raise ValueError("Unsupported type conversion detected in FieldArray: {}".format(*type_set))

    @staticmethod
    def _map_to_np_type(basic_type):
        type_mapping = {int: np.int64, float: np.float64, str: np.str}
        return type_mapping[basic_type]

    def __repr__(self):
        return "FieldArray {}: {}".format(self.name, self.content.__repr__())

    def append(self, val):
        """Add a new item to the tail of FieldArray.

        :param val: int, float, str, or a list of them.
        """
        val_type = type(val)
        if val_type is int and self.pytype is float:
            # up-cast the appended value
            val = float(val)
        elif val_type is float and self.pytype is int:
            # up-cast all other values in the content
            for idx, _ in enumerate(self.content):
                self.content[idx] = float(self.content[idx])
            self.pytype = float
            self.dtype = self._map_to_np_type(self.pytype)
        elif val_type is list:
            if len(val) == 0:
                raise ValueError("Cannot append an empty list.")
            else:
                if type(val[0]) != self.pytype:
                    raise ValueError(
                        "Cannot append a list of {}-type value into a {}-tpye FieldArray.".
                            format(type(val[0]), self.pytype))
        elif val_type != self.pytype:
            raise ValueError("Cannot append a {}-type value into a {}-tpye FieldArray.".format(val_type, self.pytype))

        self.content.append(val)

    def __getitem__(self, indices):
        return self.get(indices)

    def __setitem__(self, idx, val):
        assert isinstance(idx, int)
        self.content[idx] = val

    def get(self, indices):
        """Fetch instances based on indices.

        :param indices: an int, or a list of int.
        :return:
        """
        # TODO: 返回行为不一致，有隐患
        if isinstance(indices, int):
            return self.content[indices]
        assert self.is_input is True or self.is_target is True
        batch_size = len(indices)
        # TODO 当这个fieldArray是seq_length这种只有一位的内容时，不需要padding，需要再讨论一下
        if not is_iterable(self.content[0]):
            array = np.array([self.content[i] for i in indices], dtype=self.dtype)
        else:
            max_len = max([len(self.content[i]) for i in indices])
            array = np.full((batch_size, max_len), self.padding_val, dtype=self.dtype)
            for i, idx in enumerate(indices):
                array[i][:len(self.content[idx])] = self.content[idx]
        return array

    def __len__(self):
        """Returns the size of FieldArray.

        :return int length:
        """
        return len(self.content)


def is_iterable(content):
    try:
        _ = (e for e in content)
    except TypeError:
        return False
    return True
