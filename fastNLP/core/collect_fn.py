import torch
import numpy as np
from .field import _get_ele_type_and_dim


def _check_type(batch_dict, fields):
    if len(fields) == 0:
        raise RuntimeError
    types = []
    dims = []
    for f in fields:
        t, d = _get_ele_type_and_dim(batch_dict[f])
        types.append(t)
        dims.append(d)
    diff_types = set(types)
    diff_dims = set(dims)
    if len(diff_types) > 1 or len(diff_dims) > 1:
        raise ValueError
    return types[0]


def batching(samples, max_len=0, padding_val=0):
    if len(samples) == 0:
        return samples
    if max_len <= 0:
        max_len = max(s.shape[0] for s in samples)
    batch = np.full((len(samples), max_len), fill_value=padding_val)
    for i, s in enumerate(samples):
        slen = min(s.shape[0], max_len)
        batch[i][:slen] = s[:slen]
    return batch


class Collector:
    def __init__(self):
        self.fns = []
        self.names = []
        self.fields_list = []
        self.is_input = []

    def add_fn(self, fn, name, fields, is_input):
        if name in self.names:
            raise ValueError("Duplicated name: {} for CollectFn: {}".format(name, fn))
        if fn.num_fields() > 0 and len(fields) != fn.num_fields():
            raise ValueError(
                "Incorrect num of fields, should be {} not {}".format(
                    fn.num_fields(), len(fields)
                ))

        self.fns.append(fn)
        self.names.append(name)
        self.fields_list.append(fields)
        self.is_input.append(is_input)

    def collect_batch(self, batch_dict):
        if len(batch_dict) == 0:
            return {}, {}
        batch_x, batch_y = {}, {}
        for fn, name, fields, is_input in zip(self.fns, self.names, self.fields_list, self.is_input):
            batch = fn.collect(batch_dict, fields)
            if is_input:
                batch_x[name] = batch
            else:
                batch_y[name] = batch
        return batch_x, batch_y


class CollectFn:
    def __init__(self):
        self.fields = []

    def collect(self, batch_dict, fields):
        raise NotImplementedError

    def num_fields(self):
        return 0

    @staticmethod
    def get_batch_size(batch_dict):
        if len(batch_dict) == 0:
            return 0
        return len(next(iter(batch_dict.values())))


class ConcatCollectFn(CollectFn):
    """
    field拼接Fn，将不同field按序拼接后，padding产生数据。所有field必须有相同的dim。

    :param pad_val: padding的数值
    :param max_len: 拼接后最大长度
    """

    def __init__(self, pad_val=0, max_len=0):
        super().__init__()
        self.pad_val = pad_val
        self.max_len = max_len

    def collect(self, batch_dict, fields):
        samples = []
        dtype = _check_type(batch_dict, fields)
        batch_size = self.get_batch_size(batch_dict)
        for i in range(batch_size):
            sample = []
            for n in fields:
                seq = batch_dict[n][i]
                if str(dtype).startswith('torch'):
                    seq = seq.numpy()
                else:
                    seq = np.array(seq, dtype=dtype)
                sample.append(seq)
            samples.append(np.concatenate(sample, axis=0))
        batch = batching(samples, max_len=self.max_len, padding_val=self.pad_val)
        if str(dtype).startswith('torch'):
            batch = torch.tensor(batch, dtype=dtype)
        return batch

    def num_fields(self):
        return 0
