"""
    This file defines Action(s) and sample methods.

"""
from collections import Counter

import numpy as np
import torch


class Action(object):
    """
        Operations shared by Trainer, Tester, and Inference.
        This is designed for reducing replicate codes.
            - make_batch: produce a min-batch of data. @staticmethod
            - pad: padding method used in sequence modeling. @staticmethod
            - mode: change network mode for either train or test. (for PyTorch) @staticmethod
        The base Action shall define operations shared by as much task-specific Actions as possible.
    """

    def __init__(self):
        super(Action, self).__init__()

    @staticmethod
    def make_batch(iterator, data, use_cuda, output_length=True, max_len=None):
        """Batch and Pad data.
        :param iterator: an iterator, (object that implements __next__ method) which returns the next sample.
        :param data: list. Each entry is a sample, which is also a list of features and label(s).
            E.g.
                [
                    [[word_11, word_12, word_13], [label_11. label_12]],  # sample 1
                    [[word_21, word_22, word_23], [label_21. label_22]],  # sample 2
                    ...
                ]
        :param use_cuda: bool
        :param output_length: whether to output the original length of the sequence before padding.
        :param max_len: int, maximum sequence length
        :return (batch_x, seq_len): tuple of two elements, if output_length is true.
                     batch_x: list. Each entry is a list of features of a sample. [batch_size, max_len]
                     seq_len: list. The length of the pre-padded sequence, if output_length is True.
                 batch_y: list. Each entry is a list of labels of a sample.  [batch_size, num_labels]

                 return batch_x and batch_y, if output_length is False
        """
        for indices in iterator:
            batch = [data[idx] for idx in indices]
            batch_x = [sample[0] for sample in batch]
            batch_y = [sample[1] for sample in batch]

            batch_x = Action.pad(batch_x)
            # pad batch_y only if it is a 2-level list
            if len(batch_y) > 0 and isinstance(batch_y[0], list):
                batch_y = Action.pad(batch_y)

            # convert list to tensor
            batch_x = convert_to_torch_tensor(batch_x, use_cuda)
            batch_y = convert_to_torch_tensor(batch_y, use_cuda)

            # trim data to max_len
            if max_len is not None and batch_x.size(1) > max_len:
                batch_x = batch_x[:, :max_len]

            if output_length:
                seq_len = [len(x) for x in batch_x]
                yield (batch_x, seq_len), batch_y
            else:
                yield batch_x, batch_y

    @staticmethod
    def pad(batch, fill=0):
        """
        Pad a batch of samples to maximum length of this batch.
        :param batch: list of list
        :param fill: word index to pad, default 0.
        :return: a padded batch
        """
        max_length = max([len(x) for x in batch])
        for idx, sample in enumerate(batch):
            if len(sample) < max_length:
                batch[idx] = sample + ([fill] * (max_length - len(sample)))
        return batch

    @staticmethod
    def mode(model, test=False):
        """
        Train mode or Test mode. This is for PyTorch currently.
        :param model:
        :param test:
        """
        if test:
            model.eval()
        else:
            model.train()


def convert_to_torch_tensor(data_list, use_cuda):
    """
    convert lists into (cuda) Tensors
    :param data_list: 2-level lists
    :param use_cuda: bool
    :param reqired_grad: bool
    :return: PyTorch Tensor of shape [batch_size, max_seq_len]
    """
    data_list = torch.Tensor(data_list).long()
    if torch.cuda.is_available() and use_cuda:
        data_list = data_list.cuda()
    return data_list


def k_means_1d(x, k, max_iter=100):
    """
    Perform k-means on 1-D data.
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
        BUCKETS = ([None] * 20)
        self.length_freq = dict(Counter([len(example) for example in data_set]))
        self.buckets = k_means_bucketing(data_set, BUCKETS)

    def __iter__(self):
        bucket_samples = self.buckets[np.random.randint(0, len(self.buckets))]
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