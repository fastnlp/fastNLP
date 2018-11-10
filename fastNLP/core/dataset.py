from fastNLP.core.fieldarray import FieldArray

_READERS = {}


def construct_dataset(sentences):
    """Construct a data set from a list of sentences.

    :param sentences: list of str
    :return dataset: a DataSet object
    """
    dataset = DataSet()
    for sentence in sentences:
        instance = Instance()
        instance['raw_sentence'] = sentence
        dataset.append(instance)
    return dataset


class DataSet(object):
    """A DataSet object is a list of Instance objects.

    """

    class DataSetIter(object):
        def __init__(self, dataset):
            self.dataset = dataset
            self.idx = -1

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
            return " ".join([repr(self.dataset[name][self.idx]) for name in self.dataset])

    def __init__(self, instance=None):
        self.field_arrays = {}
        if instance is not None:
            self._convert_ins(instance)

    def __contains__(self, item):
        return item in self.field_arrays

    def __iter__(self):
        return self.DataSetIter(self)

    def _convert_ins(self, ins_list):
        if isinstance(ins_list, list):
            for ins in ins_list:
                self.append(ins)
        else:
            self.append(ins_list)

    def append(self, ins):
        # no field
        if len(self.field_arrays) == 0:
            for name, field in ins.fields.items():
                self.field_arrays[name] = FieldArray(name, [field])
        else:
            assert len(self.field_arrays) == len(ins.fields)
            for name, field in ins.fields.items():
                assert name in self.field_arrays
                self.field_arrays[name].append(field)

    def add_field(self, name, fields):
        if len(self.field_arrays) != 0:
            assert len(self) == len(fields)
        self.field_arrays[name] = FieldArray(name, fields)

    def delete_field(self, name):
        self.field_arrays.pop(name)

    def get_fields(self):
        return self.field_arrays

    def __getitem__(self, name):
        return self.field_arrays[name]

    def __len__(self):
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


if __name__ == '__main__':
    from fastNLP.core.instance import Instance

    ins = Instance(test='test0')
    dataset = DataSet([ins])
    for _iter in dataset:
        print(_iter['test'])
        _iter['test'] = 'abc'
        print(_iter['test'])
    print(dataset.field_arrays)
