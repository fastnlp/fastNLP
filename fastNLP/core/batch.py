"""
batch 模块实现了 fastNLP 所需的 Batch 类。

"""
__all__ = [
    "Batch"
]

import atexit
from queue import Empty, Full

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data

from .sampler import RandomSampler
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

    def __getitem__(self, idx: int):
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
                    data = f.padder(vlist, field_name=n, field_ele_dtype=f.dtype)
                    if not self.as_numpy:
                        data = _to_tensor(data, f.dtype)
                    batch_dict[n] = data
            return batch_dict

        return (indices,
                pad_batch(batch_x, self.inputs),
                pad_batch(batch_y, self.targets))


class Batch:
    def __init__(self, dataset, batch_size, sampler=None, buffer_size=0, as_numpy=False,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, **kwargs):

        dataset_getter = DataSetGetter(dataset, as_numpy)
        self.buffer_size = buffer_size
        self.cur_batch_indices = None
        self.num_batches = len(dataset) // batch_size + int(len(dataset) % batch_size != 0)
        shuffle = isinstance(sampler, RandomSampler)
        self.dataiter = torch.utils.data.DataLoader(
            dataset=dataset_getter, batch_size=batch_size, shuffle=shuffle,
            collate_fn=dataset_getter.collate_fn,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)

    def __iter__(self):
        for indices, batch_x, batch_y in self.dataiter:
            self.cur_batch_indices = indices
            yield batch_x, batch_y

    def get_batch_indices(self):
        return self.cur_batch_indices

    def __len__(self):
        return self.num_batches


class Batch1(object):
    """
    别名：:class:`fastNLP.Batch` :class:`fastNLP.core.batch.Batch`

    Batch 用于从 `DataSet` 中按一定的顺序, 依次按 ``batch_size`` 的大小将数据取出，
    组成 `x` 和 `y`::

        batch = Batch(data_set, batch_size=16, sampler=SequentialSampler())
        num_batch = len(batch)
        for batch_x, batch_y in batch:
            # do stuff ...

    :param dataset: :class:`~fastNLP.DataSet` 对象, 数据集
    :param int batch_size: 取出的batch大小
    :param sampler: 规定使用的 :class:`~fastNLP.Sampler` 方式. 若为 ``None`` , 使用 :class:`~fastNLP.RandomSampler`.
    
        Default: ``None``
    :param bool as_numpy: 若为 ``True`` , 输出batch为 numpy.array. 否则为 :class:`torch.Tensor`.
    
        Default: ``False``
    :param bool prefetch: 若为 ``True`` 使用多进程预先取出下一batch.
    
        Default: ``False``
    """
    
    def __init__(self, dataset, batch_size, sampler=None, as_numpy=False, prefetch=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if sampler is None:
            sampler = RandomSampler()
        self.sampler = sampler
        self.as_numpy = as_numpy
        self.idx_list = None
        self.curidx = 0
        self.num_batches = len(dataset) // batch_size + int(len(dataset) % batch_size != 0)
        self.cur_batch_indices = None
        self.prefetch = prefetch
        self.lengths = 0
    
    def fetch_one(self):
        if self.curidx >= len(self.idx_list):
            return None
        else:
            endidx = min(self.curidx + self.batch_size, len(self.idx_list))
            batch_x, batch_y = {}, {}
            
            indices = self.idx_list[self.curidx:endidx]
            self.cur_batch_indices = indices
            
            for field_name, field in self.dataset.get_all_fields().items():
                if field.is_target or field.is_input:
                    batch = field.get(indices)
                    if not self.as_numpy and field.padder is not None:
                        batch = _to_tensor(batch, field.dtype)
                    if field.is_target:
                        batch_y[field_name] = batch
                    if field.is_input:
                        batch_x[field_name] = batch
            
            self.curidx = endidx
            return batch_x, batch_y
    
    def __iter__(self):
        """
        Iterate on dataset, fetch batch data. Fetch process don't block the iterate process
        :return:
        """
        if self.prefetch:
            return self._run_batch_iter(self)
        
        def batch_iter():
            self.init_iter()
            while 1:
                res = self.fetch_one()
                if res is None:
                    break
                yield res
        
        return batch_iter()
    
    def init_iter(self):
        self.idx_list = self.sampler(self.dataset)
        self.curidx = 0
        self.lengths = self.dataset.get_length()
    
    def __len__(self):
        return self.num_batches
    
    def get_batch_indices(self):
        """
        取得当前batch在DataSet中所在的index下标序列

        :return list(int) indexes: 下标序列
        """
        return self.cur_batch_indices
    
    @staticmethod
    def _run_fetch(batch, q):
        try:
            global _python_is_exit
            batch.init_iter()
            # print('start fetch')
            while 1:
                res = batch.fetch_one()
                # print('fetch one')
                while 1:
                    try:
                        q.put(res, timeout=3)
                        break
                    except Full:
                        if _python_is_exit:
                            return
                if res is None:
                    # print('fetch done, waiting processing')
                    break
            # print('fetch exit')
        except Exception as e:
            q.put(e)
        finally:
            q.join()
    
    @staticmethod
    def _run_batch_iter(batch):
        q = mp.JoinableQueue(maxsize=10)
        fetch_p = mp.Process(target=Batch._run_fetch, args=(batch, q))
        fetch_p.daemon = True
        fetch_p.start()
        # print('fork fetch process')
        while 1:
            try:
                res = q.get(timeout=1)
                q.task_done()
                # print('get fetched')
                if res is None:
                    break
                elif isinstance(res, Exception):
                    raise res
                yield res
            except Empty as e:
                if fetch_p.is_alive():
                    continue
                else:
                    break
        fetch_p.terminate()
        fetch_p.join()
        # print('iter done')


def _to_tensor(batch, dtype):
    try:
        if dtype in (int, np.int8, np.int16, np.int32, np.int64):
            batch = torch.LongTensor(batch)
        if dtype in (float, np.float32, np.float64):
            batch = torch.FloatTensor(batch)
    except:
        pass
    return batch
