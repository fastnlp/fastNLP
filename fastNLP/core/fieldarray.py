import numpy as np


class FieldArray(object):
    """FieldArray is the collection of Instances of the same Field.
    It is the basic element of DataSet class.

    """

    def __init__(self, name, content, padding_val=0, is_target=False, is_input=False):
        """

        :param str name: the name of the FieldArray
        :param list content: a list of int, float, str or np.ndarray, or a list of list of one.
        :param int padding_val: the integer for padding. Default: 0.
        :param bool is_target: If True, this FieldArray is used to compute loss.
        :param bool is_input: If True, this FieldArray is used to the model input.
        """
        self.name = name
        if isinstance(content, list):
            content = content
        elif isinstance(content, np.ndarray):
            content = content.tolist()
        else:
            raise TypeError("content in FieldArray can only be list or numpy.ndarray, got {}.".format(type(content)))
        self.content = content
        self.padding_val = padding_val
        self.is_target = is_target
        self.is_input = is_input

        self.BASIC_TYPES = (int, float, str, np.ndarray)
        self.is_2d_list = False
        self.pytype = self._type_detection(content)
        self.dtype = self._map_to_np_type(self.pytype)

    def _type_detection(self, content):
        """

        :param content: a list of int, float, str or np.ndarray, or a list of list of one.
        :return type: one of int, float, str, np.ndarray

        """
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], list):
            # content is a 2-D list
            type_set = set([self._type_detection(x) for x in content])
            if len(type_set) > 1:
                raise RuntimeError("Cannot create FieldArray with more than one type. Provided {}".format(type_set))
            self.is_2d_list = True
            return type_set.pop()

        elif isinstance(content, list):
            # content is a 1-D list
            if len(content) == 0:
                raise RuntimeError("Cannot create FieldArray with an empty list.")
            type_set = set([type(item) for item in content])

            if len(type_set) == 1 and tuple(type_set)[0] in self.BASIC_TYPES:
                return type_set.pop()
            elif len(type_set) == 2 and float in type_set and int in type_set:
                # up-cast int to float
                return float
            else:
                raise RuntimeError("Cannot create FieldArray with type {}".format(*type_set))
        else:
            raise RuntimeError("Cannot create FieldArray with type {}".format(type(content)))

    @staticmethod
    def _map_to_np_type(basic_type):
        type_mapping = {int: np.int64, float: np.float64, str: np.str, np.ndarray: np.ndarray}
        return type_mapping[basic_type]

    def __repr__(self):
        return "FieldArray {}: {}".format(self.name, self.content.__repr__())

    def append(self, val):
        """Add a new item to the tail of FieldArray.

        :param val: int, float, str, or a list of one.
        """
        val_type = type(val)
        if val_type == list:  # shape check
            if self.is_2d_list is False:
                raise RuntimeError("Cannot append a list into a 1-D FieldArray. Please provide an element.")
            if len(val) == 0:
                raise RuntimeError("Cannot append an empty list.")
            val_list_type = [type(_) for _ in val]  # type check
            if len(val_list_type) == 2 and int in val_list_type and float in val_list_type:
                # up-cast int to float
                val_type = float
            elif len(val_list_type) == 1:
                val_type = val_list_type[0]
            else:
                raise RuntimeError("Cannot append a list of {}".format(val_list_type))
        else:
            if self.is_2d_list is True:
                raise RuntimeError("Cannot append a non-list into a 2-D list. Please provide a list.")
        if val_type == float and self.pytype == int:
            # up-cast
            self.pytype = float
            self.dtype = self._map_to_np_type(self.pytype)
        elif val_type == int and self.pytype == float:
            pass
        elif val_type == self.pytype:
            pass
        else:
            raise RuntimeError("Cannot append type {} into type {}".format(val_type, self.pytype))
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
