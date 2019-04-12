import numpy as np
import torch
import atexit

from fastNLP.core.sampler import RandomSampler
import torch.multiprocessing as mp

_python_is_exit = False
def _set_python_is_exit():
    global _python_is_exit
    _python_is_exit = True
atexit.register(_set_python_is_exit)

class Batch(object):
    """Batch is an iterable object which iterates over mini-batches.

        Example::

            for batch_x, batch_y in Batch(data_set, batch_size=16, sampler=SequentialSampler()):
                # ...

    :param DataSet dataset: a DataSet object
    :param int batch_size: the size of the batch
    :param Sampler sampler: a Sampler object
    :param bool as_numpy: If True, return Numpy array. Otherwise, return torch tensors.
    :param bool prefetch: If True, use multiprocessing to fetch next batch when training.
    :param str or torch.device device: the batch's device, if as_numpy is True, device is ignored.
    """

    def __init__(self, dataset, batch_size, sampler=RandomSampler(), as_numpy=False, prefetch=False):
        self.dataset = dataset
        self.batch_size = batch_size
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
                        batch = to_tensor(batch, field.dtype)
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
            return run_batch_iter(self)
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
        return self.cur_batch_indices


def to_tensor(batch, dtype):
    try:
        if dtype in (int, np.int8, np.int16, np.int32, np.int64):
            batch = torch.LongTensor(batch)
        if dtype in (float, np.float32, np.float64):
            batch = torch.FloatTensor(batch)
    except:
        pass
    return batch


def run_fetch(batch, q):
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
            except Exception as e:
                if _python_is_exit:
                    return
        if res is None:
            # print('fetch done, waiting processing')
            q.join()
            break
    # print('fetch exit')


def run_batch_iter(batch):
    q = mp.JoinableQueue(maxsize=10)
    fetch_p = mp.Process(target=run_fetch, args=(batch, q))
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
            yield res
        except Exception as e:
            if fetch_p.is_alive():
                continue
            else:
                break
    fetch_p.terminate()
    fetch_p.join()
    # print('iter done')

