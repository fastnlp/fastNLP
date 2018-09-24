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


class BaseSampler(object):
    """The base class of all samplers.

        Sub-classes must implement the __call__ method.
        __call__ takes a DataSet object and returns a list of int - the sampling indices.
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


def simple_sort_bucketing(lengths):
    """

    :param lengths: list of int, the lengths of all examples.
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
    lengths_mapping = [(idx, length) for idx, length in enumerate(lengths)]
    sorted_lengths = sorted(lengths_mapping, key=lambda x: x[1])
    # TODO: need to return buckets
    return [idx for idx, _ in sorted_lengths]

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


def k_means_bucketing(lengths, buckets):
    """Assign all instances into possible buckets using k-means, such that instances in the same bucket have similar lengths.

    :param lengths: list of int, the length of all samples.
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
    _, assignments = k_means_1d(lengths, num_buckets)

    for idx, bucket_id in enumerate(assignments):
        if buckets[bucket_id] is None or lengths[idx] <= buckets[bucket_id]:
            bucket_data[bucket_id].append(idx)
    return bucket_data


class BucketSampler(BaseSampler):
    """Partition all samples into multiple buckets, each of which contains sentences of approximately the same length.
    In sampling, first random choose a bucket. Then sample data from it.
    The number of buckets is decided dynamically by the variance of sentence lengths.

    """

    def __call__(self, data_set, batch_size, num_buckets):
        return self._process(data_set, batch_size, num_buckets)

    def _process(self, data_set, batch_size, num_buckets, use_kmeans=False):
        """

        :param data_set: a DataSet object
        :param batch_size: int
        :param num_buckets: int, number of buckets for grouping these sequences.
        :param use_kmeans: bool, whether to use k-means to create buckets.

        """
        buckets = ([None] * num_buckets)
        if use_kmeans is True:
            buckets = k_means_bucketing(data_set, buckets)
        else:
            buckets = simple_sort_bucketing(data_set)
        index_list = []
        for _ in range(len(data_set) // batch_size):
            chosen_bucket = buckets[np.random.randint(0, len(buckets))]
            np.random.shuffle(chosen_bucket)
            index_list += [idx for idx in chosen_bucket[:batch_size]]
        return index_list
