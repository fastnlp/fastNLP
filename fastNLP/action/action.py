import numpy as np


class Action(object):
    """
        base class for Trainer and Tester
    """

    def __init__(self):
        super(Action, self).__init__()


class BaseSampler(object):
    """
        Base class for all samplers.
    """

    def __init__(self, data_set):
        self.data_set_length = len(data_set)

    def __len__(self):
        return self.data_set_length

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(BaseSampler):
    """
    Sample data in the original order.
    """

    def __init__(self, data_set):
        super(SequentialSampler, self).__init__(data_set)

    def __iter__(self):
        return iter(range(self.data_set_length))


class RandomSampler(BaseSampler):
    """
    Sample data in random permutation order.
    """

    def __init__(self, data_set):
        super(RandomSampler, self).__init__(data_set)

    def __iter__(self):
        return iter(np.random.permutation(self.data_set_length))


class Batchifier(object):
    """
    Wrap random or sequential sampler to generate a mini-batch.
    """

    def __init__(self, sampler, batch_size, drop_last=True):
        super(Batchifier, self).__init__()
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
        if len(batch) < self.batch_size and self.drop_last is False:
            yield batch
