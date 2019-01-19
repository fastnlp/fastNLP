import numpy as np
import torch

from fastNLP.core.sampler import RandomSampler
import torch.multiprocessing as mp

class Batch(object):
    """Batch is an iterable object which iterates over mini-batches.

        Example::

            for batch_x, batch_y in Batch(data_set, batch_size=16, sampler=SequentialSampler()):
                # ...

    :param DataSet dataset: a DataSet object
    :param int batch_size: the size of the batch
    :param Sampler sampler: a Sampler object
    :param bool as_numpy: If True, return Numpy array. Otherwise, return torch tensors.

    """

    def __init__(self, dataset, batch_size, sampler=RandomSampler(), as_numpy=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.as_numpy = as_numpy
        self.idx_list = None
        self.curidx = 0
        self.num_batches = len(dataset) // batch_size + int(len(dataset) % batch_size != 0)
        self.cur_batch_indices = None

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
        return run_batch_iter(self)

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
    batch.idx_list = batch.sampler(batch.dataset)
    batch.curidx = 0
    batch.lengths = batch.dataset.get_length()
    # print('start fetch')
    while 1:
        res = batch.fetch_one()
        # print('fetch one')
        q.put(res)
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
        res = q.get()
        q.task_done()
        # print('get fetched')
        if res is None:
            break
        yield res
    fetch_p.terminate()
    fetch_p.join()
    # print('iter done')

