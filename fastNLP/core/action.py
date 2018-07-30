from collections import Counter

import numpy as np


class Action(object):
    """
        base class for Trainer and Tester
    """

    def __init__(self):
        super(Action, self).__init__()


def k_means_1d(x, k, max_iter=100):
    """

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
    """
    :param all_inst: 3-level list
                [
                    [[word_11, word_12, word_13], [label_11. label_12]],  # sample 1
                    [[word_21, word_22, word_23], [label_21. label_22]],  # sample 2
                    ...
                ]
    :param buckets: list of int. The length of the list is the number of buckets. Each integer is the maximum length
        threshold for each bucket (This is usually None.).
    :return data: 2-level list
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


class BucketSampler(BaseSampler):
    """
    Partition all samples into multiple buckets, each of which contains sentences of approximately the same length.
    In sampling, first random choose a bucket. Then sample data from it.
    The number of buckets is decided dynamically by the variance of sentence lengths.
    """

    def __init__(self, data_set):
        super(BucketSampler, self).__init__(data_set)
        BUCKETS = ([None] * 10)
        self.length_freq = dict(Counter([len(example) for example in data_set]))
        self.buckets = k_means_bucketing(data_set, BUCKETS)

    def __iter__(self):
        bucket_samples = self.buckets[np.random.randint(0, len(self.buckets) + 1)]
        np.random.shuffle(bucket_samples)
        return iter(bucket_samples)


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
        if 0 < len(batch) < self.batch_size and self.drop_last is False:
            yield batch
