"""
batch 模块实现了 fastNLP 所需的 Batch 类。

"""
__all__ = [
    "BatchIter",
    "DataSetIter",
    "TorchLoaderIter",
]

import atexit
from queue import Empty, Full

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data
from numbers import Number

from .sampler import SequentialSampler
from .dataset import DataSet

_python_is_exit = False


def _set_python_is_exit():
    global _python_is_exit
    _python_is_exit = True


atexit.register(_set_python_is_exit)


class DataSetGetter:
    def __init__(self, dataset: DataSet, as_numpy=False):
        self.dataset = dataset
        self.inputs = {n: f for n, f in dataset.get_all_fields().items() if f.is_input}
        self.targets = {n: f for n, f in dataset.get_all_fields().items() if f.is_target}
        self.as_numpy = as_numpy
        self.idx_list = list(range(len(dataset)))

    def __getitem__(self, idx: int):
        # mapping idx to sampled idx
        idx = self.idx_list[idx]
        inputs = {n:f.get(idx) for n, f in self.inputs.items()}
        targets = {n:f.get(idx) for n, f in self.targets.items()}
        return idx, inputs, targets

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch: list):
        batch_x = {n:[] for n in self.inputs.keys()}
        batch_y = {n:[] for n in self.targets.keys()}
        indices = []
        for idx, x, y in batch:
            indices.append(idx)
            for n, v in x.items():
                batch_x[n].append(v)
            for n, v in y.items():
                batch_y[n].append(v)

        def pad_batch(batch_dict, field_array):
            for n, vlist in batch_dict.items():
                f = field_array[n]
                if f.padder is None:
                    batch_dict[n] = np.array(vlist)
                else:
                    data = f.pad(vlist)
                    if not self.as_numpy:
                        data, flag = _to_tensor(data, f.dtype)
                    batch_dict[n] = data
            return batch_dict

        return (indices,
                pad_batch(batch_x, self.inputs),
                pad_batch(batch_y, self.targets))

    def set_idx_list(self, idx_list):
        if len(idx_list) != len(self.idx_list):
            raise ValueError
        self.idx_list = idx_list

    def __getattr__(self, item):
        if hasattr(self.dataset, item):
            return getattr(self.dataset, item)
        else:
            raise AttributeError("'DataSetGetter' object has no attribute '{}'".format(item))


class SamplerAdapter(torch.utils.data.Sampler):
    def __init__(self, sampler, dataset):
        self.sampler = sampler
        self.dataset = dataset

    def __iter__(self):
        return iter(self.sampler(self.dataset))


class BatchIter:
    def __init__(self):
        self.dataiter = None
        self.num_batches = None
        self.cur_batch_indices = None
        self.batch_size = None

    def init_iter(self):
        pass

    @staticmethod
    def get_num_batches(num_samples, batch_size, drop_last):
        num_batches = num_samples // batch_size
        if not drop_last and (num_samples % batch_size > 0):
            num_batches += 1
        return num_batches

    def __iter__(self):
        self.init_iter()
        for indices, batch_x, batch_y in self.dataiter:
            self.cur_batch_indices = indices
            yield batch_x, batch_y

    def get_batch_indices(self):
        return self.cur_batch_indices

    def __len__(self):
        return self.num_batches

    @property
    def dataset(self):
        return self.dataiter.dataset


class DataSetIter(BatchIter):
    def __init__(self, dataset, batch_size=1, sampler=None, as_numpy=False,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super().__init__()
        assert isinstance(dataset, DataSet)
        sampler = SamplerAdapter(sampler=sampler or SequentialSampler(), dataset=dataset)
        dataset = DataSetGetter(dataset, as_numpy)
        collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
        self.dataiter = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, sampler=sampler,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)
        self.num_batches = self.get_num_batches(len(dataset), batch_size, drop_last)
        self.batch_size = batch_size


class TorchLoaderIter(BatchIter):
    def __init__(self, dataset):
        super().__init__()
        assert isinstance(dataset, torch.utils.data.DataLoader)
        self.dataiter = dataset
        self.num_batches = self.get_num_batches(len(dataset), dataset.batch_size, dataset.drop_last)
        self.batch_size = dataset.batch_size


class OnlineDataGettter:
    # TODO
    pass


class OnlineDataIter(BatchIter):
    # TODO
    def __init__(self, dataset, batch_size=1, buffer_size=10000, sampler=None, as_numpy=False,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, **kwargs):
        super().__init__()


def _to_tensor(batch, field_dtype):
    try:
        if field_dtype is not None \
                and issubclass(field_dtype, Number) \
                and not isinstance(batch, torch.Tensor):
            if issubclass(batch.dtype.type, np.floating):
                new_batch = torch.as_tensor(batch).float()  # 默认使用float32
            else:
                new_batch = torch.as_tensor(batch)  # 复用内存地址，避免复制
            return new_batch, True
        else:
            return batch, False
    except:
        return batch, False
