from collections import Counter

import numpy as np
import torch


def convert_to_torch_tensor(data_list, use_cuda):
    """Convert lists into (cuda) Tensors.

    :param data_list: 2-level lists
    :param use_cuda: bool, whether to use GPU or not
    :return data_list: PyTorch Tensor of shape [batch_size, max_seq_len]
    """
    data_list = torch.Tensor(data_list).long()
    if torch.cuda.is_available() and use_cuda:
        data_list = data_list.cuda()
    return data_list


def k_means_1d(x, k, max_iter=100):
    """Perform k-means on 1-D data.

    :param x: list of int, representing points in 1-D.
    :param k: the number of clusters required.
    :param max_iter: maximum iteration
    :return centroids: numpy array, centroids of the k clusters
            assignment: numpy array, 1-D, the bucket id assigned to each example.
    """
    sorted_x = sorted(list(set(x)))
    if len(sorted_x) < k:
        raise ValueError("too few buckets")
    gap = len(sorted_x) / k

    centroids = np.array([sorted_x[int(x * gap)] for x in range(k)])
    assign = None

    for i in range(max_iter):
        # Cluster Assignment step
        assign = np.array([np.argmin([np.absolute(x_i - x) for x in centroids]) for x_i in x])
        # Move centroids step
        new_centroids = np.array([x[assign == k].mean() for k in range(k)])
        if (new_centroids == centroids).all():
            centroids = new_centroids
            break
        centroids = new_centroids
    return np.array(centroids), assign


def k_means_bucketing(all_inst, buckets):
    """Assign all instances into possible buckets using k-means, such that instances in the same bucket have similar lengths.

    :param all_inst: 3-level list
            E.g. ::

                [
                    [[word_11, word_12, word_13], [label_11. label_12]],  # sample 1
                    [[word_21, word_22, word_23], [label_21. label_22]],  # sample 2
                    ...
                ]

    :param buckets: list of int. The length of the list is the number of buckets. Each integer is the maximum length
        threshold for each bucket (This is usually None.).
    :return data: 2-level list
            ::

                [
                    [index_11, index_12, ...],  # bucket 1
                    [index_21, index_22, ...],  # bucket 2
                    ...
                ]

    """
    bucket_data = [[] for _ in buckets]
    num_buckets = len(buckets)
    lengths = np.array([len(inst[0]) for inst in all_inst])
    _, assignments = k_means_1d(lengths, num_buckets)

    for idx, bucket_id in enumerate(assignments):
        if buckets[bucket_id] is None or lengths[idx] <= buckets[bucket_id]:
            bucket_data[bucket_id].append(idx)
    return bucket_data


class BaseSampler(object):
    """The base class of all samplers.

    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class SequentialSampler(BaseSampler):
    """Sample data in the original order.

    """

    def __call__(self, data_set):
        return list(range(len(data_set)))


class RandomSampler(BaseSampler):
    """Sample data in random permutation order.

    """

    def __call__(self, data_set):
        return list(np.random.permutation(len(data_set)))



class Batchifier(object):
    """Wrap random or sequential sampler to generate a mini-batch.

    """

    def __init__(self, sampler, batch_size, drop_last=True):
        """

        :param sampler: a Sampler object
        :param batch_size: int, the size of the mini-batch
        :param drop_last: bool, whether to drop the last examples that are not enough to make a mini-batch.

        """
        super(Batchifier, self).__init__()
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for example in self.sampler:
            batch.append(example)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if 0 < len(batch) < self.batch_size and self.drop_last is False:
            yield batch


class BucketBatchifier(Batchifier):
    """Partition all samples into multiple buckets, each of which contains sentences of approximately the same length.
    In sampling, first random choose a bucket. Then sample data from it.
    The number of buckets is decided dynamically by the variance of sentence lengths.
    TODO: merge it into Batch
    """

    def __init__(self, data_set, batch_size, num_buckets, drop_last=True, sampler=None):
        """

        :param data_set: three-level list, shape [num_samples, 2]
        :param batch_size: int
        :param num_buckets: int, number of buckets for grouping these sequences.
        :param drop_last: bool, useless currently.
        :param sampler: Sampler, useless currently.

        """
        super(BucketBatchifier, self).__init__(sampler, batch_size, drop_last)
        buckets = ([None] * num_buckets)
        self.data = data_set
        self.batch_size = batch_size
        self.length_freq = dict(Counter([len(example) for example in data_set]))
        self.buckets = k_means_bucketing(data_set, buckets)

    def __iter__(self):
        """Make a min-batch of data."""
        for _ in range(len(self.data) // self.batch_size):
            bucket_samples = self.buckets[np.random.randint(0, len(self.buckets))]
            np.random.shuffle(bucket_samples)
            yield [self.data[idx] for idx in bucket_samples[:batch_size]]


if __name__ == "__main__":
    import random

    data = [[[y] * random.randint(0, 50), [y]] for y in range(500)]
    batch_size = 8
    iterator = iter(BucketBatchifier(data, batch_size, num_buckets=5))
    for d in iterator:
        print("\nbatch:")
        for dd in d:
            print(len(dd[0]), end=" ")
