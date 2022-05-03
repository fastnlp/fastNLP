import numpy as np
import random


class NormalSampler:
    def __init__(self, num_of_data=1000, shuffle=False):
        self._num_of_data = num_of_data
        self._data = list(range(num_of_data))
        if shuffle:
            random.shuffle(self._data)
        self.shuffle = shuffle
        self._index = 0
        self.need_reinitialize = False

    def __iter__(self):
        if self.need_reinitialize:
            self._index = 0
            if self.shuffle:
                random.shuffle(self._data)
        else:
            self.need_reinitialize = True

        return self

    def __next__(self):
        if self._index >= self._num_of_data:
            raise StopIteration
        _data = self._data[self._index]
        self._index += 1
        return _data

    def __len__(self):
        return self._num_of_data


class NormalBatchSampler:
    def __init__(self, sampler, batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class RandomDataset:
    def __init__(self, num_data=10):
        self.data = np.random.rand(num_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]



