r"""undocumented"""
from builtins import sorted

import torch
import numpy as np
from .field import _get_ele_type_and_dim
from .utils import logger
from copy import deepcopy


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


class Collater:
    r"""
    辅助DataSet管理collate_fn的类

    """
    def __init__(self):
        self.collate_fns = {}

    def add_fn(self, fn, name=None):
        r"""
        向collater新增一个collate_fn函数

        :param callable fn:
        :param str,int name:
        :return:
        """
        if name in self.collate_fns:
            logger.warn(f"collate_fn:{name} will be overwritten.")
        if name is None:
            name = len(self.collate_fns)
        self.collate_fns[name] = fn

    def is_empty(self):
        r"""
        返回是否包含collate_fn

        :return:
        """
        return len(self.collate_fns) == 0

    def delete_fn(self, name=None):
        r"""
        删除collate_fn

        :param str,int name: 如果为None就删除最近加入的collate_fn
        :return:
        """
        if not self.is_empty():
            if name in self.collate_fns:
                self.collate_fns.pop(name)
            elif name is None:
                last_key = list(self.collate_fns.keys())[0]
                self.collate_fns.pop(last_key)

    def collate_batch(self, ins_list):
        bx, by = {}, {}
        for name, fn in self.collate_fns.items():
            try:
                batch_x, batch_y = fn(ins_list)
            except BaseException as e:
                logger.error(f"Exception:`{e}` happens when call collate_fn:`{name}`.")
                raise e
            bx.update(batch_x)
            by.update(batch_y)
        return bx, by

    def copy_from(self, col):
        assert isinstance(col, Collater)
        new_col = Collater()
        new_col.collate_fns = deepcopy(col.collate_fns)
        return new_col


class ConcatCollateFn:
    r"""
    field拼接collate_fn，将不同field按序拼接后，padding产生数据。

    :param List[str] inputs: 将哪些field的数据拼接起来, 目前仅支持1d的field
    :param str output: 拼接后的field名称
    :param pad_val: padding的数值
    :param max_len: 拼接后最大长度
    :param is_input: 是否将生成的output设置为input
    :param is_target: 是否将生成的output设置为target
    """

    def __init__(self, inputs, output, pad_val=0, max_len=0, is_input=True, is_target=False):
        super().__init__()
        assert isinstance(inputs, list)
        self.inputs = inputs
        self.output = output
        self.pad_val = pad_val
        self.max_len = max_len
        self.is_input = is_input
        self.is_target = is_target

    @staticmethod
    def _to_numpy(seq):
        if torch.is_tensor(seq):
            return seq.numpy()
        else:
            return np.array(seq)

    def __call__(self, ins_list):
        samples = []
        for i, ins in ins_list:
            sample = []
            for input_name in self.inputs:
                sample.append(self._to_numpy(ins[input_name]))
            samples.append(np.concatenate(sample, axis=0))
        batch = batching(samples, max_len=self.max_len, padding_val=self.pad_val)
        b_x, b_y = {}, {}
        if self.is_input:
            b_x[self.output] = batch
        if self.is_target:
            b_y[self.output] = batch

        return b_x, b_y
