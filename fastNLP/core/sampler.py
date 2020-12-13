r"""
sampler 子类实现了 fastNLP 所需的各种采样器。
"""
__all__ = [
    "Sampler",
    "BucketSampler",
    "SequentialSampler",
    "RandomSampler",
    "SortedSampler",
    "ConstantTokenNumSampler"
]

from itertools import chain

import numpy as np


class Sampler(object):
    r"""
    `Sampler` 类的基类. 规定以何种顺序取出data中的元素

    子类必须实现 ``__call__`` 方法. 输入 `DataSet` 对象, 返回其中元素的下标序列
    """
    
    def __call__(self, data_set):
        r"""
        :param DataSet data_set: `DataSet` 对象, 需要Sample的数据
        :return result: list(int) 其中元素的下标序列, ``data_set`` 中元素会按 ``result`` 中顺序取出
        """
        raise NotImplementedError


class SequentialSampler(Sampler):
    r"""
    顺序取出元素的 `Sampler`

    """
    
    def __call__(self, data_set):
        return list(range(len(data_set)))


class RandomSampler(Sampler):
    r"""
    随机化取元素的 `Sampler`

    """
    
    def __call__(self, data_set):
        return list(np.random.permutation(len(data_set)))


class BucketSampler(Sampler):
    r"""
    带Bucket的 `Random Sampler`. 可以随机地取出长度相似的元素
    """
    
    def __init__(self, num_buckets=10, batch_size=None, seq_len_field_name='seq_len'):
        r"""
        
        :param int num_buckets: bucket的数量
        :param int batch_size: batch的大小. 默认为None，Trainer/Tester在调用BucketSampler时，会将该值正确设置，如果是非
            Trainer/Tester场景使用，需要显示传递该值
        :param str seq_len_field_name: 对应序列长度的 `field` 的名字
        """
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.seq_len_field_name = seq_len_field_name

    def set_batch_size(self, batch_size):
        r"""

        :param int batch_size: 每个batch的大小
        :return:
        """
        self.batch_size = batch_size

    def __call__(self, data_set):
        if self.batch_size is None:
            raise RuntimeError("batch_size is None.")
        seq_lens = data_set.get_all_fields()[self.seq_len_field_name].content
        total_sample_num = len(seq_lens)
        
        bucket_indexes = []
        assert total_sample_num >= self.num_buckets, "The number of samples is smaller than the number of buckets."
        num_sample_per_bucket = total_sample_num // self.num_buckets
        for i in range(self.num_buckets):
            bucket_indexes.append([num_sample_per_bucket * i, num_sample_per_bucket * (i + 1)])
        bucket_indexes[-1][1] = total_sample_num
        
        sorted_seq_lens = list(sorted([(idx, seq_len) for
                                       idx, seq_len in zip(range(total_sample_num), seq_lens)],
                                      key=lambda x: x[1]))
        
        batchs = []
        
        left_init_indexes = []
        for b_idx in range(self.num_buckets):
            start_idx = bucket_indexes[b_idx][0]
            end_idx = bucket_indexes[b_idx][1]
            sorted_bucket_seq_lens = sorted_seq_lens[start_idx:end_idx]
            left_init_indexes.extend([tup[0] for tup in sorted_bucket_seq_lens])
            num_batch_per_bucket = len(left_init_indexes) // self.batch_size
            np.random.shuffle(left_init_indexes)
            for i in range(num_batch_per_bucket):
                batchs.append(left_init_indexes[i * self.batch_size:(i + 1) * self.batch_size])
            left_init_indexes = left_init_indexes[num_batch_per_bucket * self.batch_size:]
        if (left_init_indexes) != 0:
            batchs.append(left_init_indexes)
        np.random.shuffle(batchs)
        
        return list(chain(*batchs))


class ConstantTokenNumSampler:
    """
    尽量保证每个batch的输入token数量是接近的。

    使用示例
    >>> # 假设已经有了tr_data并有一个field叫做seq_len保存了每个instance的token数量
    >>> from fastNLP import DataSetIter, Trainer
    >>> sampler = BatchSampler(tr_data.get_field('seq_len').content, max_token=4096)
    >>> tr_iter = DataSetIter(tr_data,
    >>>                     batch_size=1, sampler=None, as_numpy=False, num_workers=0, pin_memory=False,
    >>>                     drop_last=False, timeout=0, worker_init_fn=None,
    >>>                     batch_sampler=sampler)
    >>>
    >>> # 直接将tr_iter传入Trainer中，此时batch_size参数的值会被忽略
    >>> trainer = Trainer(tr_iter, model, optimizer=optimizer, loss=TranslationLoss(),
    >>>             batch_size=1, sampler=None, drop_last=False, update_every=1)

    """
    def __init__(self, seq_len, max_token=4096, max_sentence=-1, need_be_multiple_of=1, num_bucket=-1):
        """

        :param List[int] seq_len: list[int], 是每个sample的长度。一般可以通过dataset.get_field('seq_len').content传入
        :param int max_token: 每个batch的最大的token数量
        :param int max_sentence: 每个batch最多多少个instance, -1表示根据max_token决定
        :param int need_be_multiple_of: 生成的batch的instance的数量需要是几的倍数，在DataParallel场景下会用到
        :param int num_bucket: 将数据按长度拆分为num_bucket个bucket，batch中的sample尽量在bucket之中进行组合，这样可以减少padding。
        """
        assert (max_sentence!=-1 and max_sentence>=need_be_multiple_of) or max_sentence<1
        assert len(seq_len)>num_bucket, "The number of samples should be larger than buckets."
        self.seq_len = seq_len
        self.max_token = max_token
        self._max_sentence = max_sentence
        self.need_be_multiple_of = need_be_multiple_of
        seq_len_indice = [(length, i) for i, length in enumerate(seq_len)]
        seq_len_indice.sort(key=lambda x: x[0])
        indice_in_buckets = []
        if num_bucket>0:
            sample_per_bucket = len(seq_len_indice)//num_bucket
            i = 0
            while len(indice_in_buckets)<len(seq_len_indice):
                indice_in_buckets.append(seq_len_indice[i*sample_per_bucket:(i+1)*sample_per_bucket])
                i += 1
        else:
            indice_in_buckets = [seq_len_indice]
        self.indice_in_buckets = indice_in_buckets
        self.get_new_order()

    @property
    def max_sentence(self):
        if self._max_sentence<1:
            return 100000000
        return self._max_sentence

    @max_sentence.setter
    def max_sentence(self, max_sentence):
        self._max_sentence = max_sentence

    def get_new_order(self):
        np.random.shuffle(self.indice_in_buckets)
        for bucket in self.indice_in_buckets:
            np.random.shuffle(bucket)
        indices = list(chain(*self.indice_in_buckets))
        batches = []
        cur_max_len = 0
        batch = []
        for length, i in indices:
            max_len = max(length, cur_max_len)
            if max_len*(len(batch)+1)>self.max_token or len(batch)>=self.max_sentence:
                left_sample = len(batch) % self.need_be_multiple_of
                add_samples = batch.copy()
                cur_max_len =length
                if left_sample!=0:
                    add_samples = add_samples[:-left_sample]
                    batch = batch[-left_sample:]
                    cur_max_len = max(cur_max_len, max(batch))
                else:
                    batch = []
                if len(add_samples)==0:
                    raise RuntimeError(f"The sample `{i}` is too long to make a batch with {self.need_be_multiple_of} samples.")
                batches.append(add_samples)
            else:
                cur_max_len = max_len
            batch.append(i)
        if batch:
            left_sample = len(batch) % self.need_be_multiple_of
            add_samples = batch.copy()
            if left_sample != 0:
                add_samples = add_samples[:-left_sample].copy()
            if add_samples:
                batches.append(add_samples)
        np.random.shuffle(batches)
        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
            yield batch
        self.get_new_order()

    def __len__(self):
        return len(self.batches)


class SortedSampler(Sampler):
    r"""
    按照sample的长度进行排序，主要在测试的时候使用，可以加速测试（因为减少了padding）
    """
    def __init__(self, seq_len_field_name='seq_len', descending=True):
        """

        :param str seq_len_field_name: 对应序列长度的 `field` 的名字
        :param bool descending: 是否降序排列
        """
        self.seq_len_field_name = seq_len_field_name
        self.descending = descending

    def __call__(self, data_set):
        seq_lens = data_set.get_field(self.seq_len_field_name).content
        orders = np.argsort(seq_lens).tolist()  # 从小到大的顺序
        if self.descending:
            orders = orders[::-1]
        return orders


def simple_sort_bucketing(lengths):
    r"""

    :param lengths: list of int, the lengths of all examples.
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
    r"""Perform k-means on 1-D data.

    :param x: list of int, representing points in 1-D.
    :param k: the number of clusters required.
    :param max_iter: maximum iteration
    :return centroids: numpy array, centroids of the k clusters
            assignment: numpy array, 1-D, the bucket id assigned to each example.
    """
    sorted_x = sorted(list(set(x)))
    x = np.array(x)
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
    r"""Assign all instances into possible buckets using k-means, such that instances in the same bucket have similar lengths.

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
