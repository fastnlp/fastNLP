"""
batch 模块实现了 fastNLP 所需的 :class:`~fastNLP.core.batch.DataSetIter` 类。

"""
__all__ = [
    "BatchIter",
    "DataSetIter",
    "TorchLoaderIter",
]

import atexit
from numbers import Number

import numpy as np
import torch
import torch.utils.data

from ._logger import logger
from .dataset import DataSet
from .sampler import SequentialSampler

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
        """

        :param batch: [[idx1, x_dict1, y_dict1], [idx2, x_dict2, y_dict2], [xx, xx, xx]]
        :return:
        """
        # TODO 支持在DataSet中定义collate_fn，因为有时候可能需要不同的field之间融合，比如BERT的场景
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
                        try:
                            data, flag = _to_tensor(data, f.dtype)
                        except TypeError as e:
                            logger.error(f"Field {n} cannot be converted to torch.tensor.")
                            raise e
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
        super().__init__(dataset)
        self.sampler = sampler
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.sampler(self.dataset))


class BatchIter:
    def __init__(self, dataset, batch_size=1, sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None):
        if not isinstance(sampler, torch.utils.data.Sampler):
            self.sampler = SamplerAdapter(sampler=sampler or SequentialSampler(), dataset=dataset)
        else:
            self.sampler = sampler
        if collate_fn is None:
            # pytoch <= 1.1 中不能设置collate_fn=None
            self.dataiter = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=batch_size, sampler=self.sampler,
                num_workers=num_workers,
                pin_memory=pin_memory, drop_last=drop_last,
                timeout=timeout, worker_init_fn=worker_init_fn)
        else:
            self.dataiter = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=batch_size, sampler=self.sampler,
                collate_fn=collate_fn, num_workers=num_workers,
                pin_memory=pin_memory, drop_last=drop_last,
                timeout=timeout, worker_init_fn=worker_init_fn)

        # 以sampler的数量为准，因为DistributedSampler的时候每个进程上并不是所有的数据都用上了
        self.num_batches = self.get_num_batches(len(self.dataiter.sampler), batch_size, drop_last)
        self.batch_size = batch_size
        self.cur_batch_indices = None

    def init_iter(self):
        pass

    @staticmethod
    def get_num_batches(num_samples, batch_size, drop_last):
        """
        计算batch的数量。

        :param int num_samples:
        :param int batch_size:
        :param bool drop_last: 如果最后一个batch没有batch_size这么多，是否就丢掉。
        :return:
        """
        num_batches = num_samples // batch_size
        if not drop_last and (num_samples % batch_size > 0):
            num_batches += 1
        return num_batches

    def get_batch_indices(self):
        """
        获取当前已经输出的batch的index。

        :return:
        """
        return self.cur_batch_indices

    def __len__(self):
        return self.num_batches

    @property
    def dataset(self):
        return self.dataiter.dataset


class DataSetIter(BatchIter):
    """
    DataSetIter 用于从 `DataSet` 中按一定的顺序, 依次按 ``batch_size`` 的大小将数据取出，
    组成 `x` 和 `y`::

        batch = DataSetIter(data_set, batch_size=16, sampler=SequentialSampler())
        num_batch = len(batch)
        for batch_x, batch_y in batch:
            # do stuff ...

    """
    def __init__(self, dataset, batch_size=1, sampler=None, as_numpy=False,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None):
        """
        
        :param dataset: :class:`~fastNLP.DataSet` 对象, 数据集
        :param int batch_size: 取出的batch大小
        :param sampler: 规定使用的 :class:`~fastNLP.Sampler` 方式. 若为 ``None`` , 使用 :class:`~fastNLP.SequentialSampler`.
    
            Default: ``None``
        :param bool as_numpy: 若为 ``True`` , 输出batch为 numpy.array. 否则为 :class:`torch.Tensor`.

            Default: ``False``
        :param int num_workers: 使用多少个进程来预处理数据
        :param bool pin_memory: 是否将产生的tensor使用pin memory, 可能会加快速度。
        :param bool drop_last: 如果最后一个batch没有batch_size这么多sample，就扔掉最后一个
        :param timeout: 生成一个batch的timeout值
        :param worker_init_fn: 在每个worker启动时调用该函数，会传入一个值，该值是worker的index。
        :param collate_fn: 用于将样本组合成batch的函数
        """
        assert isinstance(dataset, DataSet)
        dataset = DataSetGetter(dataset, as_numpy)
        collate_fn = dataset.collate_fn if collate_fn is None else collate_fn
        super().__init__(
            dataset=dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )

    def __iter__(self):
        self.init_iter()
        for indices, batch_x, batch_y in self.dataiter:
            self.cur_batch_indices = indices
            yield batch_x, batch_y


class TorchLoaderIter(BatchIter):
    """
    与DataSetIter类似，但用于pytorch的DataSet对象。
    通过使用TorchLoaderIter封装pytorch的DataSet，然后将其传入到Trainer中。

    """
    def __init__(self, dataset, batch_size=1, sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None):
        """

        :param dataset: :class:`~fastNLP.DataSet` 对象, 数据集
        :param int batch_size: 取出的batch大小
        :param sampler: 规定使用的 :class:`~fastNLP.Sampler` 方式. 若为 ``None`` , 使用 :class:`~fastNLP.SequentialSampler`.

            Default: ``None``
        :param int num_workers: 使用多少个进程来预处理数据
        :param bool pin_memory: 是否将产生的tensor使用pin memory, 可能会加快速度。
        :param bool drop_last: 如果最后一个batch没有batch_size这么多sample，就扔掉最后一个
        :param timeout: 生成一个batch的timeout值
        :param worker_init_fn: 在每个worker启动时调用该函数，会传入一个值，该值是worker的index。
        :param collate_fn: 用于将样本组合成batch的函数"""
        assert len(dataset) > 0
        ins = dataset[0]
        assert len(ins) == 2 and \
               isinstance(ins[0], dict) and \
               isinstance(ins[1], dict), 'DataSet should return two dict, as X and Y'

        super().__init__(
            dataset=dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )

    def __iter__(self):
        self.init_iter()
        for batch_x, batch_y in self.dataiter:
            self.cur_batch_indices = None
            yield batch_x, batch_y


def _to_tensor(batch, field_dtype):
    """

    :param batch: np.array()
    :param field_dtype: 数据类型
    :return: batch, flag. 如果传入的数据支持转为tensor，返回的batch就是tensor，且flag为True；如果传入的数据不支持转为tensor，
        返回的batch就是原来的数据，且flag为False
    """
    try:
        if field_dtype is not None and isinstance(field_dtype, type)\
                and issubclass(field_dtype, Number) \
                and not isinstance(batch, torch.Tensor):
            if issubclass(batch.dtype.type, np.floating):
                new_batch = torch.as_tensor(batch).float()  # 默认使用float32
            elif issubclass(batch.dtype.type, np.integer):
                new_batch = torch.as_tensor(batch).long()  # 复用内存地址，避免复制
            else:
                new_batch = torch.as_tensor(batch)
            return new_batch, True
        else:
            return batch, False
    except Exception as e:
        raise e
