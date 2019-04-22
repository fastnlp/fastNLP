import _pickle as pickle

import numpy as np

from fastNLP.core.fieldarray import AutoPadder
from fastNLP.core.fieldarray import FieldArray
from fastNLP.core.instance import Instance
from fastNLP.core.utils import get_func_signature


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
                data_set.add_field(name=field.name, fields=field.content[idx], padder=field.padder,
                                   is_input=field.is_input, is_target=field.is_target, ignore_type=field.ignore_type)
            return data_set
        elif isinstance(idx, str):
            if idx not in self:
                raise KeyError("No such field called {} in DataSet.".format(idx))
            return self.field_arrays[idx]
        else:
            raise KeyError("Unrecognized type {} for idx in __getitem__ method".format(type(idx)))

    def __getattr__(self, item):
        # Not tested. Don't use !!
        if item == "field_arrays":
            raise AttributeError
        if isinstance(item, str) and item in self.field_arrays:
            return self.field_arrays[item]

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
                field = field.tolist() if isinstance(field, np.ndarray) else field
                self.field_arrays[name] = FieldArray(name, [field])  # 第一个样本，必须用list包装起来
        else:
            if len(self.field_arrays) != len(ins.fields):
                raise ValueError(
                    "DataSet object has {} fields, but attempt to append an Instance object with {} fields."
                        .format(len(self.field_arrays), len(ins.fields)))
            for name, field in ins.fields.items():
                assert name in self.field_arrays
                self.field_arrays[name].append(field)

    def add_field(self, name, fields, padder=None, is_input=False, is_target=False, ignore_type=False):
        """Add a new field to the DataSet.
        
        :param str name: the name of the field.
        :param fields: a list of int, float, or other objects.
        :param padder: PadBase对象，如何对该Field进行padding。如果为None则使用
        :param bool is_input: whether this field is model input.
        :param bool is_target: whether this field is label or target.
        :param bool ignore_type: If True, do not perform type check. (Default: False)
        """
        if padder is None:
            padder = AutoPadder(pad_val=0)

        if len(self.field_arrays) != 0:
            if len(self) != len(fields):
                raise RuntimeError(f"The field to append must have the same size as dataset. "
                                   f"Dataset size {len(self)} != field size {len(fields)}")
        self.field_arrays[name] = FieldArray(name, fields, is_target=is_target, is_input=is_input,
                                             padder=padder, ignore_type=ignore_type)

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
        """将field_names的target设置为flag状态
        Example::

            dataset.set_target('labels', 'seq_len')  # 将labels和seq_len这两个field的target属性设置为True
            dataset.set_target('labels', 'seq_lens', flag=False) # 将labels和seq_len的target属性设置为False

        :param field_names: str, field的名称
        :param flag: bool, 将field_name的target状态设置为flag
        """
        assert isinstance(flag, bool), "Only bool type supported."
        for name in field_names:
            if name in self.field_arrays:
                self.field_arrays[name].is_target = flag
            else:
                raise KeyError("{} is not a valid field name.".format(name))

    def set_input(self, *field_names, flag=True):
        """将field_name的input设置为flag状态
        Example::

            dataset.set_input('words', 'seq_len')   # 将words和seq_len这两个field的input属性设置为True
            dataset.set_input('words', flag=False)  # 将words这个field的input属性设置为False

        :param field_names: str, field的名称
        :param flag: bool, 将field_name的input状态设置为flag
        """
        for name in field_names:
            if name in self.field_arrays:
                self.field_arrays[name].is_input = flag
            else:
                raise KeyError("{} is not a valid field name.".format(name))

    def set_ignore_type(self, *field_names, flag=True):
        """将field_names的ignore_type设置为flag状态

        :param field_names: str, field的名称
        :param flag: bool,
        :return:
        """
        assert isinstance(flag, bool), "Only bool type supported."
        for name in field_names:
            if name in self.field_arrays:
                self.field_arrays[name].ignore_type = flag
            else:
                raise KeyError("{} is not a valid field name.".format(name))

    def set_padder(self, field_name, padder):
        """为field_name设置padder
        Example::

            from fastNLP import EngChar2DPadder
            padder = EngChar2DPadder()
            dataset.set_padder('chars', padder)  # 则chars这个field会使用EngChar2DPadder进行pad操作

        :param field_name: str, 设置field的padding方式为padder
        :param padder: (None, PadderBase). 设置为None即删除padder, 即对该field不进行padding操作.
        :return:
        """
        if field_name not in self.field_arrays:
            raise KeyError("There is no field named {}.".format(field_name))
        self.field_arrays[field_name].set_padder(padder)

    def set_pad_val(self, field_name, pad_val):
        """为某个field设置对应的pad_val.

        :param field_name: str，修改该field的pad_val
        :param pad_val: int，该field的padder会以pad_val作为padding index
        :return:
        """
        if field_name not in self.field_arrays:
            raise KeyError("There is no field named {}.".format(field_name))
        self.field_arrays[field_name].set_pad_val(pad_val)

    def get_input_name(self):
        """返回所有is_input被设置为True的field名称

        :return list, 里面的元素为被设置为input的field名称
        """
        return [name for name, field in self.field_arrays.items() if field.is_input]

    def get_target_name(self):
        """返回所有is_target被设置为True的field名称

        :return list, 里面的元素为被设置为target的field名称
        """
        return [name for name, field in self.field_arrays.items() if field.is_target]

    def apply_field(self, func, field_name, new_field_name=None, **kwargs):
        """将DataSet中的每个instance中的`field_name`这个field传给func，并获取它的返回值.

        :param func: Callable, input是instance的`field_name`这个field.
        :param field_name: str, 传入func的是哪个field.
        :param new_field_name: (str, None). 如果不是None，将func的返回值放入这个名为`new_field_name`的新field中，如果名称与已有
                               的field相同，则覆盖之前的field.
        :param **kwargs: 合法的参数有以下三个
                        (1) is_input: bool, 如果为True则将`new_field_name`这个field设置为input
                        (2) is_target: bool, 如果为True则将`new_field_name`这个field设置为target
                        (3) ignore_type: bool, 如果为True则将`new_field_name`这个field的ignore_type设置为true, 忽略其类型
        :return: List[], 里面的元素为func的返回值，所以list长度为DataSet的长度

        """
        assert len(self)!=0, "Null DataSet cannot use apply()."
        if field_name not in self:
            raise KeyError("DataSet has no field named `{}`.".format(field_name))
        results = []
        idx = -1
        try:
            for idx, ins in enumerate(self._inner_iter()):
                results.append(func(ins[field_name]))
        except Exception as e:
            if idx!=-1:
                print("Exception happens at the `{}`th instance.".format(idx))
            raise e
        if not (new_field_name is None) and len(list(filter(lambda x: x is not None, results))) == 0:  # all None
            raise ValueError("{} always return None.".format(get_func_signature(func=func)))

        if new_field_name is not None:
            self._add_apply_field(results, new_field_name, kwargs)

        return results

    def _add_apply_field(self, results, new_field_name, kwargs):
        """将results作为加入到新的field中，field名称为new_field_name

        :param results: List[], 一般是apply*()之后的结果
        :param new_field_name: str, 新加入的field的名称
        :param kwargs: dict, 用户apply*()时传入的自定义参数
        :return:
        """
        extra_param = {}
        if 'is_input' in kwargs:
            extra_param['is_input'] = kwargs['is_input']
        if 'is_target' in kwargs:
            extra_param['is_target'] = kwargs['is_target']
        if 'ignore_type' in kwargs:
            extra_param['ignore_type'] = kwargs['ignore_type']
        if new_field_name in self.field_arrays:
            # overwrite the field, keep same attributes
            old_field = self.field_arrays[new_field_name]
            if 'is_input' not in extra_param:
                extra_param['is_input'] = old_field.is_input
            if 'is_target' not in extra_param:
                extra_param['is_target'] = old_field.is_target
            if 'ignore_type' not in extra_param:
                extra_param['ignore_type'] = old_field.ignore_type
            self.add_field(name=new_field_name, fields=results, is_input=extra_param["is_input"],
                           is_target=extra_param["is_target"], ignore_type=extra_param['ignore_type'])
        else:
            self.add_field(name=new_field_name, fields=results, is_input=extra_param.get("is_input", None),
                           is_target=extra_param.get("is_target", None),
                           ignore_type=extra_param.get("ignore_type", False))

    def apply(self, func, new_field_name=None, **kwargs):
        """将DataSet中每个instance传入到func中，并获取它的返回值.

        :param func: Callable, 参数是DataSet中的instance
        :param new_field_name: (None, str). (1) None, 不创建新的field; (2) str，将func的返回值放入这个名为
                              `new_field_name`的新field中，如果名称与已有的field相同，则覆盖之前的field;
        :param kwargs: 合法的参数有以下三个
                        (1) is_input: bool, 如果为True则将`new_field_name`的field设置为input
                        (2) is_target: bool, 如果为True则将`new_field_name`的field设置为target
                        (3) ignore_type: bool, 如果为True则将`new_field_name`的field的ignore_type设置为true, 忽略其类型
        :return: List[], 里面的元素为func的返回值，所以list长度为DataSet的长度
        """
        assert len(self)!=0, "Null DataSet cannot use apply()."
        idx = -1
        try:
            results = []
            for idx, ins in enumerate(self._inner_iter()):
                results.append(func(ins))
        except Exception as e:
            if idx!=-1:
                print("Exception happens at the `{}`th instance.".format(idx))
            raise e
        # results = [func(ins) for ins in self._inner_iter()]
        if not (new_field_name is None) and len(list(filter(lambda x: x is not None, results))) == 0:  # all None
            raise ValueError("{} always return None.".format(get_func_signature(func=func)))

        if new_field_name is not None:
            self._add_apply_field(results, new_field_name, kwargs)

        return results

    def drop(self, func, inplace=True):
        """func接受一个instance，返回bool值，返回值为True时，该instance会被删除。

        :param func: Callable, 接受一个instance作为参数，返回bool值。为True时删除该instance
        :param inplace: bool, 是否在当前DataSet中直接删除instance。如果为False，返回值为一个删除了相应instance的新的DataSet

        :return: DataSet.
        """
        if inplace:
            results = [ins for ins in self._inner_iter() if not func(ins)]
            for name, old_field in self.field_arrays.items():
                self.field_arrays[name].content = [ins[name] for ins in results]
            return self
        else:
            results = [ins for ins in self if not func(ins)]
            data = DataSet(results)
            for field_name, field in self.field_arrays.items():
                data.field_arrays[field_name].to(field)
            return data

    def split(self, ratio):
        """将DataSet按照ratio的比例拆分，返回两个DataSet

        :param ratio: float, 0<ratio<1, 返回的第一个DataSet拥有ratio这么多数据，第二个DataSet拥有(1-ratio)这么多数据
        :return (DataSet, DataSet)
        """
        assert isinstance(ratio, float)
        assert 0 < ratio < 1
        all_indices = [_ for _ in range(len(self))]
        np.random.shuffle(all_indices)
        split = int(ratio * len(self))
        dev_indices = all_indices[:split]
        train_indices = all_indices[split:]
        dev_set = DataSet()
        train_set = DataSet()
        for idx in dev_indices:
            dev_set.append(self[idx])
        for idx in train_indices:
            train_set.append(self[idx])
        for field_name in self.field_arrays:
            train_set.field_arrays[field_name].to(self.field_arrays[field_name])
            dev_set.field_arrays[field_name].to(self.field_arrays[field_name])

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
        with open(csv_path, "r", encoding='utf-8') as f:
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

    def save(self, path):
        """保存DataSet.

        :param path: str, 将DataSet存在哪个路径
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """从保存的DataSet pickle路径中读取DataSet

        :param path: str, 读取路径
        :return  DataSet:
        """
        with open(path, 'rb') as f:
            d = pickle.load(f)
            assert isinstance(d, DataSet), "The object is not DataSet, but {}.".format(type(d))
        return d


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
