import random
import sys
from collections import defaultdict
from copy import deepcopy
import numpy as np

from fastNLP.core.field import TextField, LabelField
from fastNLP.core.instance import Instance
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.fieldarray import FieldArray

_READERS = {}

class DataSet(object):
    """A DataSet object is a list of Instance objects.

    """

    def __init__(self, instance=None):
        if instance is not None:
            self._convert_ins(instance)
        else:
            self.field_arrays = {}

    def _convert_ins(self, ins_list):
        if isinstance(ins_list, list):
            for ins in ins_list:
                self.append(ins)
        else:
            self.append(ins)

    def append(self, ins):
        # no field
        if len(self.field_arrays) == 0:
            for name, field in ins.field.items():
                self.field_arrays[name] = FieldArray(name, [field])
        else:
            assert len(self.field_arrays) == len(ins.field)
            for name, field in ins.field.items():
                assert name in self.field_arrays
                self.field_arrays[name].append(field)

    def get_length(self):
        """Fetch lengths of all fields in all instances in a dataset.

        :return lengths: dict of (str: list). The str is the field name.
                The list contains lengths of this field in all instances.

        """
        pass

    def shuffle(self):
        pass

    def split(self, ratio, shuffle=True):
        """Train/dev splitting

        :param ratio: float, between 0 and 1. The ratio of development set in origin data set.
        :param shuffle: bool, whether shuffle the data set before splitting. Default: True.
        :return train_set: a DataSet object, representing the training set
                dev_set: a DataSet object, representing the validation set

        """
        pass

    def rename_field(self, old_name, new_name):
        """rename a field
        """
        if old_name in self.field_arrays:
            self.field_arrays[new_name] = self.field_arrays.pop(old_name)
        else:
            raise KeyError
        return self

    def set_is_target(self, **fields):
        """Change the flag of `is_target` for all instance. For fields not set here, leave their `is_target` unchanged.

        :param key-value pairs for field-name and `is_target` value(True, False).
        """
        for name, val in fields.items():
            if name in self.field_arrays:
                assert isinstance(val, bool)
                self.field_arrays[name].is_target = val
            else:
                raise KeyError
        return self

    def set_need_tensor(self, **kwargs):
        for name, val in kwargs.items():
            if name in self.field_arrays:
                assert isinstance(val, bool)
                self.field_arrays[name].need_tensor = val
            else:
                raise KeyError
        return self

    def __getattribute__(self, name):
        if name in _READERS:
            # add read_*data() support
            def _read(*args, **kwargs):
                data = _READERS[name]().load(*args, **kwargs)
                self.extend(data)
                return self
            return _read
        else:
            return object.__getattribute__(self, name)

    @classmethod
    def set_reader(cls, method_name):
        """decorator to add dataloader support
        """
        assert isinstance(method_name, str)
        def wrapper(read_cls):
            _READERS[method_name] = read_cls
            return read_cls
        return wrapper
