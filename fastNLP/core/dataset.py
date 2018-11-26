import numpy as np

from fastNLP.core.fieldarray import FieldArray
from fastNLP.core.instance import Instance

_READERS = {}


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


class DataSet(object):
    """DataSet is the collection of examples.
    DataSet provides instance-level interface. You can append and access an instance of the DataSet.
    However, it stores data in a different way: Field-first, Instance-second.

    """

    class Instance(object):
        def __init__(self, dataset, idx=-1, **fields):
            self.dataset = dataset
            self.idx = idx
            self.fields = fields

        def __next__(self):
            self.idx += 1
            if self.idx >= len(self.dataset):
                raise StopIteration
            return self

        def add_field(self, field_name, field):
            """Add a new field to the instance.

            :param field_name: str, the name of the field.
            :param field:
            """
            self.fields[field_name] = field

        def __getitem__(self, name):
            return self.dataset[name][self.idx]

        def __setitem__(self, name, val):
            if name not in self.dataset:
                new_fields = [None] * len(self.dataset)
                self.dataset.add_field(name, new_fields)
            self.dataset[name][self.idx] = val

        def __repr__(self):
            return "\n".join(['{}: {}'.format(name, repr(self.dataset[name][self.idx])) for name
                              in self.dataset.get_fields().keys()])

    def __init__(self, data=None):
        """

        :param data: a dict or a list. If it is a dict, the key is the name of a field and the value is the field.
                All values must be of the same length.
                If it is a list, it must be a list of Instance objects.
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
        return self.Instance(self)

    def _convert_ins(self, ins_list):
        if isinstance(ins_list, list):
            for ins in ins_list:
                self.append(ins)
        else:
            self.append(ins_list)

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
            assert len(self) == len(fields)
        self.field_arrays[name] = FieldArray(name, fields, padding_val=padding_val, is_target=is_target,
                                             is_input=is_input)

    def delete_field(self, name):
        """Delete a field based on the field name.

        :param str name: the name of the field to be deleted.
        """
        self.field_arrays.pop(name)

    def get_fields(self):
        """Return all the fields with their names.

        :return dict field_arrays: the internal data structure of DataSet.
        """
        return self.field_arrays

    def __getitem__(self, idx):
        """

        :param idx: can be int, slice, or str.
        :return: If `idx` is int, return an Instance object.
                If `idx` is slice, return a DataSet object.
                If `idx` is str, it must be a field name, return the field.

        """
        if isinstance(idx, int):
            return self.Instance(self, idx, **{name: self.field_arrays[name][idx] for name in self.field_arrays})
        elif isinstance(idx, slice):
            data_set = DataSet()
            for field in self.field_arrays.values():
                data_set.add_field(name=field.name,
                                   fields=field.content[idx],
                                   padding_val=field.padding_val,
                                   is_input=field.is_input,
                                   is_target=field.is_target)
            return data_set
        elif isinstance(idx, str):
            return self.field_arrays[idx]
        else:
            raise KeyError("Unrecognized type {} for idx in __getitem__ method".format(type(idx)))

    def __len__(self):
        if len(self.field_arrays) == 0:
            return 0
        field = iter(self.field_arrays.values()).__next__()
        return len(field)

    def get_length(self):
        """The same as __len__

        """
        return len(self)

    def rename_field(self, old_name, new_name):
        """rename a field
        """
        if old_name in self.field_arrays:
            self.field_arrays[new_name] = self.field_arrays.pop(old_name)
        else:
            raise KeyError("{} is not a valid name. ".format(old_name))

    def set_target(self, **fields):
        """Change the flag of `is_target` for all instance. For fields not set here, leave their `is_target` unchanged.

        :param key-value pairs for field-name and `is_target` value(True, False).
        """
        for name, val in fields.items():
            if name in self.field_arrays:
                assert isinstance(val, bool)
                self.field_arrays[name].is_target = val
            else:
                raise KeyError("{} is not a valid field name.".format(name))
        return self

    def set_input(self, **fields):
        for name, val in fields.items():
            if name in self.field_arrays:
                assert isinstance(val, bool)
                self.field_arrays[name].is_input = val
            else:
                raise KeyError("{} is not a valid field name.".format(name))
        return self

    def get_input_name(self):
        return [name for name, field in self.field_arrays.items() if field.is_input]

    def get_target_name(self):
        return [name for name, field in self.field_arrays.items() if field.is_target]

    def __getattr__(self, item):
        # block infinite recursion for copy, pickle
        if item == '__setstate__':
            raise AttributeError(item)
        try:
            return self.field_arrays.__getitem__(item)
        except KeyError:
            pass
        try:
            reader_cls = _READERS[item]

            # add read_*data() support
            def _read(*args, **kwargs):
                data = reader_cls().load(*args, **kwargs)
                self.extend(data)
                return self

            return _read
        except KeyError:
            raise AttributeError('{} does not exist.'.format(item))

    @classmethod
    def set_reader(cls, method_name):
        """decorator to add dataloader support
        """
        assert isinstance(method_name, str)

        def wrapper(read_cls):
            _READERS[method_name] = read_cls
            return read_cls

        return wrapper

    def apply(self, func, new_field_name=None):
        """Apply a function to every instance of the DataSet.

        :param func: a function that takes an instance as input.
        :param str new_field_name: If not None, results of the function will be stored as a new field.
        :return results: returned values of the function over all instances.
        """
        results = [func(ins) for ins in self]
        if new_field_name is not None:
            if new_field_name in self.field_arrays:
                # overwrite the field, keep same attributes
                old_field = self.field_arrays[new_field_name]
                self.add_field(name=new_field_name,
                               fields=results,
                               padding_val=old_field.padding_val,
                               is_input=old_field.is_input,
                               is_target=old_field.is_target)
            else:
                self.add_field(name=new_field_name, fields=results)
        else:
            return results

    def split(self, dev_ratio):
        """Split the dataset into training and development(validation) set.

        :param float dev_ratio: the ratio of test set in all data.
        :return DataSet train_set: the training set
                DataSet dev_set: the development set
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
        return train_set, dev_set

    @classmethod
    def read_csv(cls, csv_path, headers=None, sep='\t'):
        with open(csv_path, 'r') as f:
            start_idx = 0
            if headers is None:
                headers = f.readline()
                headers = headers.split(sep)
                start_idx += 1
            else:
                assert isinstance(headers, list), "headers should be list, not {}.".format(type(headers))
            _dict = {}
            for col in headers:
                _dict[col] = []
            for line_idx, line in enumerate(f, start_idx):
                contents = line.split(sep)
                assert len(contents)==len(headers), "Line {} has {} parts, while header has {}."\
                    .format(line_idx, len(contents), len(headers))
                for header, content in zip(headers, contents):
                    _dict[header].append(content)
        return cls(_dict)