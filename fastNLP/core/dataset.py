import _pickle as pickle

import numpy as np

from fastNLP.core.fieldarray import FieldArray
from fastNLP.core.instance import Instance
from fastNLP.core.utils import get_func_signature
from fastNLP.io.base_loader import DataLoaderRegister


class DataSet(object):
    """DataSet is the collection of examples.
    DataSet provides instance-level interface. You can append and access an instance of the DataSet.
    However, it stores data in a different way: Field-first, Instance-second.

    """

    def __init__(self, data=None):
        """

        :param data: a dict or a list.
                If `data` is a dict, the key is the name of a FieldArray and the value is the FieldArray. All values
                must be of the same length.
                If `data` is a list, it must be a list of Instance objects.
        """
        self.field_arrays = {}
        if data is not None:
            if isinstance(data, dict):
                length_set = set()
                for key, value in data.items():
                    length_set.add(len(value))
                assert len(length_set) == 1, "Arrays must all be same length."
                for key, value in data.items():
                    self.add_field(name=key, fields=value)
            elif isinstance(data, list):
                for ins in data:
                    assert isinstance(ins, Instance), "Must be Instance type, not {}.".format(type(ins))
                    self.append(ins)

            else:
                raise ValueError("data only be dict or list type.")

    def __contains__(self, item):
        return item in self.field_arrays

    def __iter__(self):
        def iter_func():
            for idx in range(len(self)):
                yield self[idx]

        return iter_func()

    def _inner_iter(self):
        class Iter_ptr:
            def __init__(self, dataset, idx):
                self.dataset = dataset
                self.idx = idx

            def __getitem__(self, item):
                assert item in self.dataset.field_arrays, "no such field:{} in Instance {}".format(item, self.dataset[
                    self.idx])
                assert self.idx < len(self.dataset.field_arrays[item]), "index:{} out of range".format(self.idx)
                return self.dataset.field_arrays[item][self.idx]

            def __repr__(self):
                return self.dataset[self.idx].__repr__()

        def inner_iter_func():
            for idx in range(len(self)):
                yield Iter_ptr(self, idx)

        return inner_iter_func()

    def __getitem__(self, idx):
        """Fetch Instance(s) at the `idx` position(s) in the dataset.
        Notice: This method returns a copy of the actual instance(s). Any change to the returned value would not modify
        the origin instance(s) of the DataSet.
        If you want to make in-place changes to all Instances, use `apply` method.

        :param idx: can be int or slice.
        :return: If `idx` is int, return an Instance object.
                If `idx` is slice, return a DataSet object.
        """
        if isinstance(idx, int):
            return Instance(**{name: self.field_arrays[name][idx] for name in self.field_arrays})
        elif isinstance(idx, slice):
            if idx.start is not None and (idx.start >= len(self) or idx.start <= -len(self)):
                raise RuntimeError(f"Start index {idx.start} out of range 0-{len(self)-1}")
            data_set = DataSet()
            for field in self.field_arrays.values():
                data_set.add_field(name=field.name,
                                   fields=field.content[idx],
                                   padding_val=field.padding_val,
                                   is_input=field.is_input,
                                   is_target=field.is_target)
            return data_set
        else:
            raise KeyError("Unrecognized type {} for idx in __getitem__ method".format(type(idx)))

    def __getattr__(self, item):
        # Not tested. Don't use !!
        if item == "field_arrays":
            raise AttributeError
        if isinstance(item, str) and item in self.field_arrays:
            return self.field_arrays[item]
        try:
            reader = DataLoaderRegister.get_reader(item)
            return reader
        except AttributeError:
            raise

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__

    def __len__(self):
        """Fetch the length of the dataset.

        :return length:
        """
        if len(self.field_arrays) == 0:
            return 0
        field = iter(self.field_arrays.values()).__next__()
        return len(field)

    def __inner_repr__(self):
        if len(self) < 20:
            return ",\n".join([ins.__repr__() for ins in self])
        else:
            return self[:5].__inner_repr__() + "\n...\n" + self[-5:].__inner_repr__()

    def __repr__(self):
        return "DataSet(" + self.__inner_repr__() + ")"

    def append(self, ins):
        """Add an instance to the DataSet.
        If the DataSet is not empty, the instance must have the same field names as the rest instances in the DataSet.

        :param ins: an Instance object

        """
        if len(self.field_arrays) == 0:
            # DataSet has no field yet
            for name, field in ins.fields.items():
                self.field_arrays[name] = FieldArray(name, [field])
        else:
            assert len(self.field_arrays) == len(ins.fields)
            for name, field in ins.fields.items():
                assert name in self.field_arrays
                self.field_arrays[name].append(field)

    def add_field(self, name, fields, padding_val=0, is_input=False, is_target=False):
        """Add a new field to the DataSet.
        
        :param str name: the name of the field.
        :param fields: a list of int, float, or other objects.
        :param int padding_val: integer for padding.
        :param bool is_input: whether this field is model input.
        :param bool is_target: whether this field is label or target.
        """
        if len(self.field_arrays) != 0:
            if len(self) != len(fields):
                raise RuntimeError(f"The field to append must have the same size as dataset. "
                                   f"Dataset size {len(self)} != field size {len(fields)}")
        self.field_arrays[name] = FieldArray(name, fields, padding_val=padding_val, is_target=is_target,
                                             is_input=is_input)

    def delete_field(self, name):
        """Delete a field based on the field name.

        :param name: the name of the field to be deleted.
        """
        self.field_arrays.pop(name)

    def get_field(self, field_name):
        if field_name not in self.field_arrays:
            raise KeyError("Field name {} not found in DataSet".format(field_name))
        return self.field_arrays[field_name]

    def get_all_fields(self):
        """Return all the fields with their names.

        :return field_arrays: the internal data structure of DataSet.
        """
        return self.field_arrays

    def get_length(self):
        """Fetch the length of the dataset.

        :return length:
        """
        return len(self)

    def rename_field(self, old_name, new_name):
        """Rename a field.

        :param str old_name:
        :param str new_name:
        """
        if old_name in self.field_arrays:
            self.field_arrays[new_name] = self.field_arrays.pop(old_name)
            self.field_arrays[new_name].name = new_name
        else:
            raise KeyError("DataSet has no field named {}.".format(old_name))

    def set_target(self, *field_names, flag=True):
        """Change the target flag of these fields.

        :param field_names: a sequence of str, indicating field names
        :param bool flag: Set these fields as target if True. Unset them if False.
        """
        for name in field_names:
            if name in self.field_arrays:
                self.field_arrays[name].is_target = flag
            else:
                raise KeyError("{} is not a valid field name.".format(name))

    def set_input(self, *field_name, flag=True):
        """Set the input flag of these fields.

        :param field_name: a sequence of str, indicating field names.
        :param bool flag: Set these fields as input if True. Unset them if False.
        """
        for name in field_name:
            if name in self.field_arrays:
                self.field_arrays[name].is_input = flag
            else:
                raise KeyError("{} is not a valid field name.".format(name))

    def get_input_name(self):
        """Get all field names with `is_input` as True.

        :return field_names: a list of str
        """
        return [name for name, field in self.field_arrays.items() if field.is_input]

    def get_target_name(self):
        """Get all field names with `is_target` as True.

        :return field_names: a list of str
        """
        return [name for name, field in self.field_arrays.items() if field.is_target]

    def apply(self, func, new_field_name=None, **kwargs):
        """Apply a function to every instance of the DataSet.

        :param func: a function that takes an instance as input.
        :param str new_field_name: If not None, results of the function will be stored as a new field.
        :param **kwargs: Accept parameters will be
            (1) is_input: boolean, will be ignored if new_field is None. If True, the new field will be as input.
            (2) is_target: boolean, will be ignored if new_field is None. If True, the new field will be as target.
        :return results: if new_field_name is not passed, returned values of the function over all instances.
        """
        results = [func(ins) for ins in self._inner_iter()]

        extra_param = {}
        if 'is_input' in kwargs:
            extra_param['is_input'] = kwargs['is_input']
        if 'is_target' in kwargs:
            extra_param['is_target'] = kwargs['is_target']
        if new_field_name is not None:
            if len(list(filter(lambda x: x is not None, results))) == 0:  # all None
                raise ValueError("{} always return None.".format(get_func_signature(func=func)))
            if new_field_name in self.field_arrays:
                # overwrite the field, keep same attributes
                old_field = self.field_arrays[new_field_name]
                if 'is_input' not in extra_param:
                    extra_param['is_input'] = old_field.is_input
                if 'is_target' not in extra_param:
                    extra_param['is_target'] = old_field.is_target
                self.add_field(name=new_field_name,
                               fields=results,
                               padding_val=old_field.padding_val,
                               **extra_param)
            else:
                self.add_field(name=new_field_name, fields=results, **extra_param)
        else:
            return results

    def drop(self, func):
        """Drop instances if a condition holds.

        :param func: a function that takes an Instance object as input, and returns bool.
            The instance will be dropped if the function returns True.

        """
        results = [ins for ins in self._inner_iter() if not func(ins)]
        for name, old_field in self.field_arrays.items():
            self.field_arrays[name].content = [ins[name] for ins in results]

    def split(self, dev_ratio):
        """Split the dataset into training and development(validation) set.

        :param float dev_ratio: the ratio of test set in all data.
        :return (train_set, dev_set):
                train_set: the training set
                dev_set: the development set
        """
        assert isinstance(dev_ratio, float)
        assert 0 < dev_ratio < 1
        all_indices = [_ for _ in range(len(self))]
        np.random.shuffle(all_indices)
        split = int(dev_ratio * len(self))
        dev_indices = all_indices[:split]
        train_indices = all_indices[split:]
        dev_set = DataSet()
        train_set = DataSet()
        for idx in dev_indices:
            dev_set.append(self[idx])
        for idx in train_indices:
            train_set.append(self[idx])
        for field_name in self.field_arrays:
            train_set.field_arrays[field_name].is_input = self.field_arrays[field_name].is_input
            train_set.field_arrays[field_name].is_target = self.field_arrays[field_name].is_target
            dev_set.field_arrays[field_name].is_input = self.field_arrays[field_name].is_input
            dev_set.field_arrays[field_name].is_target = self.field_arrays[field_name].is_target

        return train_set, dev_set

    @classmethod
    def read_csv(cls, csv_path, headers=None, sep=",", dropna=True):
        """Load data from a CSV file and return a DataSet object.

        :param str csv_path: path to the CSV file
        :param List[str] or Tuple[str] headers: headers of the CSV file
        :param str sep: delimiter in CSV file. Default: ","
        :param bool dropna: If True, drop rows that have less entries than headers.
        :return dataset: the read data set

        """
        with open(csv_path, "r") as f:
            start_idx = 0
            if headers is None:
                headers = f.readline().rstrip('\r\n')
                headers = headers.split(sep)
                start_idx += 1
            else:
                assert isinstance(headers, (list, tuple)), "headers should be list or tuple, not {}.".format(
                    type(headers))
            _dict = {}
            for col in headers:
                _dict[col] = []
            for line_idx, line in enumerate(f, start_idx):
                contents = line.rstrip('\r\n').split(sep)
                if len(contents) != len(headers):
                    if dropna:
                        continue
                    else:
                        # TODO change error type
                        raise ValueError("Line {} has {} parts, while header has {} parts." \
                                         .format(line_idx, len(contents), len(headers)))
                for header, content in zip(headers, contents):
                    _dict[header].append(content)
        return cls(_dict)

    # def read_pos(self):
    #     return DataLoaderRegister.get_reader('read_pos')

    def save(self, path):
        """Save the DataSet object as pickle.

        :param str path: the path to the pickle
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load a DataSet object from pickle.

        :param str path: the path to the pickle
        :return data_set:
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


def construct_dataset(sentences):
    """Construct a data set from a list of sentences.

    :param sentences: list of list of str
    :return dataset: a DataSet object
    """
    dataset = DataSet()
    for sentence in sentences:
        instance = Instance()
        instance['raw_sentence'] = sentence
        dataset.append(instance)
    return dataset
