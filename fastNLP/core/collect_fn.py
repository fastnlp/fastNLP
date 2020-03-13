from builtins import sorted

import torch
import numpy as np
from .field import _get_ele_type_and_dim
from collections import defaultdict


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
        self.fns = {}
        self.input2fn = defaultdict(list)
        self.output2fn = defaultdict(list)
        self.fn2input = {}
        self.fn2output = {}

    def add_fn(self, fn, inputs, outputs, is_input, is_target):
        for name in outputs:
            if name in self.output2fn:
                raise ValueError("Duplicated name: {} for CollectFn: {}".format(name, fn))

        if fn.num_inputs() > 0 and len(inputs) != fn.num_inputs():
            raise ValueError(
                "Incorrect num of inputs, should be {} not {}".format(
                    fn.num_inputs(), len(inputs)
                ))

        if fn.num_outputs() > 0 and len(outputs) != fn.num_outputs():
            raise ValueError("Incorrect num of inputs, should be {} not {}".format(
                    fn.num_outputs(), len(outputs)))

        self.fns[fn] = {'is_input': is_input, 'is_target': is_target}
        for i, field in enumerate(inputs):
            self.input2fn[field].append((fn, i))
        for i, name in enumerate(outputs):
            self.output2fn[name].append((fn, i))

    def _rebuild_fn2io(self):
        def transpose(name2fn):
            fn2names = defaultdict(list)
            for name, vlist in name2fn.items():
                for fn, i in vlist:
                    fn2names[fn].append((name, i))
            for fn, vlist in fn2names.items():
                vlist = sorted(vlist, key=lambda x: x[1])
                fn2names[fn] = [name for name, i in vlist]
            return fn2names

        self.fn2input = transpose(self.input2fn)
        self.fn2output = transpose(self.output2fn)

    def _clear_fn2io(self):
        self.fn2input.clear()
        self.fn2output.clear()

    def collect_batch(self, ins_list):
        if len(ins_list) == 0:
            return {}, {}

        if len(self.fn2output) == 0:
            self._rebuild_fn2io()

        bx = {}
        by = {}
        for fn, attr in self.fns.items():
            inputs = self.fn2input.get(fn, None)
            outputs = self.fn2output.get(fn, None)
            res = fn.collect(ins_list, inputs, outputs)
            if attr.get('is_input', False):
                bx.update(res)
            if attr.get('is_target', False):
                by.update(res)
        return bx, by

    def rename_field(self, old_f, new_f):
        if new_f in self.input2fn:
            # name conflict
            raise ValueError
        if old_f not in self.input2fn:
            # renamed field not affect collectors
            return
        self.input2fn[new_f] = self.input2fn[old_f]
        self._clear_fn2io()

    def drop_field(self, f):
        if f in self.input2fn:
            raise ValueError

    def outputs(self):
        return self.output2fn.keys()


class CollectFn:
    def __init__(self):
        self.fields = []

    def collect(self, ins_list, inputs, outputs):
        raise NotImplementedError

    def num_inputs(self):
        return 0

    def num_outputs(self):
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

    @staticmethod
    def _to_numpy(seq):
        if torch.is_tensor(seq):
            return seq.numpy()
        else:
            return np.array(seq)

    def collect(self, ins_list, inputs, outputs):
        samples = []
        for i, ins in ins_list:
            sample = []
            for i in inputs:
                sample.append(self._to_numpy(ins[i]))
            samples.append(np.concatenate(sample, axis=0))
        seq_len = [s.shape[0] for s in samples]
        batch = batching(samples, max_len=self.max_len, padding_val=self.pad_val)
        o1, o2 = outputs
        return {o1: batch, o2: seq_len}

    def num_inputs(self):
        return 0

    def num_outputs(self):
        # (concat_words, seq_len)
        return 2
