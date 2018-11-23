import numpy as np

from fastNLP.core.fieldarray import FieldArray

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
        def __init__(self, dataset, idx=-1):
            self.dataset = dataset
            self.idx = idx

        def __next__(self):
            self.idx += 1
            if self.idx >= len(self.dataset):
                raise StopIteration
            return self

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

    def add_field(self, name, fields, padding_val=0, need_tensor=False, is_target=False):
        """
        
        :param name:
        :param fields:
        :param padding_val:
        :param need_tensor:
        :param is_target:
        :return:
        """
        if len(self.field_arrays) != 0:
            assert len(self) == len(fields)
        self.field_arrays[name] = FieldArray(name, fields,
                                             padding_val=padding_val,
                                             need_tensor=need_tensor,
                                             is_target=is_target)

    def delete_field(self, name):
        self.field_arrays.pop(name)

    def get_fields(self):
        return self.field_arrays

    def __getitem__(self, name):
        if isinstance(name, int):
            return self.Instance(self, idx=name)
        elif isinstance(name, slice):
            ds = DataSet()
            for field in self.field_arrays.values():
                ds.add_field(name=field.name,
                             fields=field.content[name],
                             padding_val=field.padding_val,
                             need_tensor=field.need_tensor,
                             is_target=field.is_target)
            return ds

        elif isinstance(name, str):
            return self.field_arrays[name]
        else:
            raise KeyError

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
                raise KeyError("{} is not a valid field name.".format(name))
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

    def __getattr__(self, item):
        if item in self.field_arrays:
            return self.field_arrays[item]
        else:
            self.__getattribute__(item)

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
        results = []
        for ins in self:
            results.append(func(ins))
        if new_field_name is not None:
            if new_field_name in self.field_arrays:
                # overwrite the field, keep same attributes
                old_field = self.field_arrays[new_field_name]
                padding_val = old_field.padding_val
                need_tensor = old_field.need_tensor
                is_target = old_field.is_target
                self.add_field(new_field_name, results, padding_val, need_tensor, is_target)
            else:
                self.add_field(new_field_name, results)
        else:
            return results

    def split(self, test_ratio):
        assert isinstance(test_ratio, float)
        all_indices = [_ for _ in range(len(self))]
        np.random.shuffle(all_indices)
        test_indices = all_indices[:int(test_ratio)]
        train_indices = all_indices[int(test_ratio):]
        test_set = DataSet()
        train_set = DataSet()
        for idx in test_indices:
            test_set.append(self[idx])
        for idx in train_indices:
            train_set.append(self[idx])
        return train_set, test_set


if __name__ == '__main__':
    from fastNLP.core.instance import Instance

    d = DataSet({'a': list('abc')})
    _ = d.a
    d.apply(lambda x: x['a'])
    print(d[1])
