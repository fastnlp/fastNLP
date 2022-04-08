r"""
sampler 子类实现了 fastNLP 所需的各种采样器。
"""

__all__ = [
    "BucketSampler",
    "SortedSampler",
    'ConstTokenNumSampler',
    "ConstantTokenNumSampler",
    "UnrepeatedDistributedSampler",
]

from itertools import chain
from typing import List, Iterable

import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    from torch.utils.data import SequentialSampler, Sampler, RandomSampler
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Sampler

# class DopedSampler(Sampler):
#     """
#     定制给MixDataLoader的BatchSampler，其功能是将传入的datasets的list列表混合采样组成一个个batch返回。
#     """
#
#     def __init__(self, dataset: Union[List, Dict], batch_size: int = None,
#                  sampler: Union[List[Sampler], Dict[str, Sampler]] = None,
#                  ds_ratio: Union[str, None, List[float], Dict[str, float]] = None, drop_last: bool = False) -> None:
#         if batch_size <= 0:
#             raise ValueError("batch_size should be a positive integer value, "
#                              "but got batch_size={}".format(batch_size))
#         if not isinstance(drop_last, bool):
#             raise ValueError("drop_last should be a boolean value, but got "
#                              "drop_last={}".format(drop_last))
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         self.ds_ratio = ds_ratio
#         if sampler is None:
#             if isinstance(dataset, List):
#                 self.sampler = [SequentialSampler(ds) for ds in dataset]
#             elif isinstance(dataset, Dict):
#                 self.sampler = {name: SequentialSampler(ds) for name, ds in dataset.items()}
#
#         elif isinstance(sampler, List):
#             if len(sampler) != len(dataset):
#                 raise ValueError("the length of sampler != the length of sampler")
#             self.sampler = sampler
#         else:
#             self.sampler = sampler
#         if ds_ratio == 'pad_to_most' or ds_ratio == 'truncate_to_least' or ds_ratio is None:
#             self.ds_ratio = ds_ratio
#         elif isinstance(ds_ratio, List):
#             if not all(item >= 0 for item in ds_ratio):
#                 raise ValueError("batch_size should be a positive integer value, "
#                                  "but got batch_size={}".format(ds_ratio))
#             self.ds_ratio = ds_ratio
#         else:
#             raise ValueError(f"{ds_ratio} must be pad_to_least or truncate_to_least or None")
#
#     def __iter__(self):
#         samplers, index = [], 0
#         if isinstance(self.sampler, List):
#             for idx, sampler in enumerate(self.sampler):
#                 samplers.append((iter(sampler), self.batch_size, index, 0, idx))
#                 index += len(sampler)
#         elif isinstance(self.sampler, Dict):
#             for name, sampler in self.sampler.items():
#                 samplers.append((iter(sampler), self.batch_size, index, 0, name))
#                 index += len(sampler)
#
#     def __len__(self):
#         lens = 0
#         max_len, ds_len = 0, 0
#         if self.ds_ratio == 'truncate_to_least':
#             if isinstance(self.sampler, List):
#                 max_len = min(len(sampler) for sampler in self.sampler)
#                 ds_len = len(self.sampler)
#             elif isinstance(self.sampler, Dict):
#                 max_len = min(len(sampler) for _, sampler in self.sampler.items())
#                 for _, _ in self.sampler.items():
#                     ds_len += 1
#
#         elif self.ds_ratio == 'pad_to_most':
#             if isinstance(self.sampler, List):
#                 max_len = max(len(sampler) for sampler in self.sampler)
#                 ds_len = len(self.sampler)
#             elif isinstance(self.sampler, Dict):
#                 max_len = max(len(sampler) for _, sampler in self.sampler.items())
#                 for _, _ in self.sampler.items():
#                     ds_len += 1
#
#         if self.ds_ratio is None:
#             if isinstance(self.sampler, List):
#                 for i in range(len(self.sampler)):
#                     sampler = self.sampler[i]
#                     if self.drop_last:
#                         lens += len(sampler) // self.batch_size
#                     else:
#                         lens += (len(sampler) + self.batch_size - 1) // self.batch_size
#             elif isinstance(self.sampler, Dict):
#                 for name, sampler in self.sampler.items():
#                     if self.drop_last:
#                         lens += len(sampler) // self.batch_size
#                     else:
#                         lens += (len(sampler) + self.batch_size - 1) // self.batch_size
#         elif self.ds_ratio == 'truncate_to_least' or self.ds_ratio == 'pad_to_most':
#             for i in range(ds_len):
#                 if self.drop_last:
#                     lens += max_len // self.batch_size
#                 else:
#                     lens += (max_len + self.batch_size - 1) // self.batch_size
#         return lens
#
#     def demo(self):
#         indexes = np.array([0]*self.batch_size + [1]*self.batch_size + [2]*self.batch_size)
#         shift = np.array([0]*self.batch_size + [len(ds1)]*self.batch_size + [len(ds1)+len(ds2)]*self.batch_size)
#         buffer = np.zeros(self.batch_size*self.num_ds, dtype=int)
#         select_sampler = np.random.randint(0, self.batch_size*self.num_ds, num_sample=self.batch_size)
#         select_indices = buffer[select_sampler] + shift[select_sampler]
#         num_1 = (indexes[select_sampler]==0).sum()
#


# class MixSequentialSampler(Sampler):
#     """
#     定制给MixDataLoader的BatchSampler，其功能是将传入的datasets的list列表顺序采样并返回index，只有处理了上一个dataset才会处理下一个。
#     """
#
#     def __init__(self, dataset: Union[List, Dict], batch_size: int = None,
#                  sampler: Union[List[Sampler], Dict[str, Sampler], None] = None,
#                  drop_last: bool = False) -> None:
#         """
#
#         :param dataset: 实现了__getitem__和__len__的数据容器列表
#         :param batch_size: 对应dataset的批次大小，可以为list或者为int，当为int时默认所有dataset
#         :param sampler: 实例化好的sampler，每个dataset对应一个sampler对象
#         :param drop_last: 是否去掉最后一个batch的数据，其长度小于batch_size
#         """
#         # 如果dataset为Dict，则其他参数如collate_fn必须为Dict或者Callable,
#         if isinstance(dataset, Dict) and isinstance(sampler, List):
#             raise ValueError(f"{sampler} must be dict")
#
#         # 判断batch_size是否大于等于0
#         if batch_size <= 0:
#             raise ValueError("batch_size should be a positive integer value, "
#                              "but got batch_size={}".format(batch_size))
#
#         if not isinstance(drop_last, bool):
#             raise ValueError("drop_last should be a boolean value, but got "
#                              "drop_last={}".format(drop_last))
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         if sampler is None:
#             if isinstance(dataset, List):
#                 self.sampler = [SequentialSampler(ds) for ds in dataset]
#             elif isinstance(dataset, Dict):
#                 self.sampler = {name: SequentialSampler(ds) for name, ds in dataset.items()}
#         elif isinstance(sampler, List):
#             if len(sampler) != len(dataset):
#                 raise ValueError("the length of sampler != the length of sampler")
#             self.sampler = sampler
#
#     def __iter__(self) -> Iterable[List[int]]:
#         """
#         按照dataset的顺序采样，打包成一个batch后返回
#         :return:
#         """
#         index = 0
#         batch = []
#         if isinstance(self. sampler, List):
#             for i in range(len(self.sampler)):
#                 sampler = self.sampler[i]
#                 for idx in sampler:
#                     batch.append(idx + index)
#                     if len(batch) == self.batch_size:
#                         yield batch
#                         batch = []
#                 if len(batch) > 0 and not self.drop_last:
#                     yield batch
#                     batch = []
#                 index += len(sampler)
#         elif isinstance(self.sampler, Dict):
#             for name, sampler in self.sampler.items():
#                 for idx in sampler:
#                     batch.append(idx + index)
#                     if len(batch) == self.batch_size:
#                         yield batch
#                         batch = []
#                 if len(batch) > 0 and not self.drop_last:
#                     yield batch
#                     batch = []
#                 index += len(sampler)
#
#     def __len__(self) -> int:
#         lens = 0
#         if isinstance(self.sampler, List):
#             for i in range(len(self.sampler)):
#                 sampler = self.sampler[i]
#                 if self.drop_last:
#                     lens += len(sampler) // self.batch_size
#                 else:
#                     lens += (len(sampler) + self.batch_size - 1) // self.batch_size
#         elif isinstance(self.sampler, Dict):
#             for _, sampler in self.sampler.items():
#                 if self.drop_last:
#                     lens += len(sampler) // self.batch_size
#                 else:
#                     lens += (len(sampler) + self.batch_size - 1) // self.batch_size
#         return lens


# class PollingSampler(Sampler):
#     """
#     定制给MixDataLoader的BatchSampler，其功能是将传入的datasets的list列表轮流采样并返回index，处理了上个dataset的一个batch后会处理下一个。
#     """
#
#     def __init__(self, dataset: Union[List, Dict], batch_size: int = 16,
#                  sampler: Union[List[Sampler], Dict[str, Sampler]] = None,
#                  drop_last: bool = False, ds_ratio="pad_to_most") -> None:
#         """
#
#         :param dataset: 实现了__getitem__和__len__的数据容器列表
#         :param batch_size: 对应dataset的批次大小，可以为list或者为int，当为int时默认所有dataset
#         :param sampler: 实例化好的sampler，每个dataset对应一个sampler对象
#         :param drop_last: 是否去掉最后一个batch的数据，其长度小于batch_size
#         :param ds_ratio: 当ds_ratio=None时候， 轮流采样dataset列表直至所有的数据集采样完；当ds_ratio='truncate_to_least'时，
#         以dataset列表最短的ds为基准，长的数据集会被截断；当ds_ratio='pad_to_most'时，以dataset列表最长ds为基准，短的数据集会被重采样
#        """
#         # 如果dataset为Dict，则其他参数如collate_fn必须为Dict或者Callable,
#         if isinstance(dataset, Dict) and isinstance(sampler, List):
#             raise ValueError(f"{sampler} must be dict")
#         if isinstance(dataset, List) and isinstance(sampler, Dict):
#             raise ValueError(f"{sampler} must be list")
#         # 判断batch_size是否大于等于0
#         if batch_size <= 0:
#             raise ValueError("batch_size should be a positive integer value, "
#                              "but got batch_size={}".format(batch_size))
#
#         if not isinstance(drop_last, bool):
#             raise ValueError("drop_last should be a boolean value, but got "
#                              "drop_last={}".format(drop_last))
#
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         if sampler is None:
#             if isinstance(dataset, List):
#                 self.sampler = [SequentialSampler(ds) for ds in dataset]
#             elif isinstance(dataset, Dict):
#                 self.sampler = {name: SequentialSampler(ds) for name, ds in dataset.items()}
#
#         elif isinstance(sampler, List):
#             if len(sampler) != len(dataset):
#                 raise ValueError("the length of sampler != the length of sampler")
#             self.sampler = sampler
#         else:
#             self.sampler = sampler
#         if ds_ratio == 'pad_to_most' or ds_ratio == 'truncate_to_least' or ds_ratio is None:
#             self.ds_ratio = ds_ratio
#         else:
#             raise ValueError(f"{ds_ratio} must be pad_to_least or truncate_to_least or None")
#
#     def __iter__(self) -> Iterable[List[int]]:
#         # index是数据集下标基址， pointer指向数据集列表的某个数据集
#         index, pointer, samplers, flag = 0, 0, [], False
#
#         if isinstance(self.sampler, List):
#             for idx, sampler in enumerate(self.sampler):
#                 samplers.append((iter(sampler), self.batch_size, index, 0, idx))
#                 index += len(sampler)
#         elif isinstance(self.sampler, Dict):
#             for name, sampler in self.sampler.items():
#                 samplers.append((iter(sampler), self.batch_size, index, 0, name))
#                 index += len(sampler)
#         if self.ds_ratio == 'pad_to_most':
#             if isinstance(self.sampler, List):
#                 limit_len = max(len(ds) for ds in self.sampler)
#             else:
#                 limit_len = max(len(ds) for _, ds in self.sampler.items())
#         elif self.ds_ratio == 'truncate_to_least':
#             if isinstance(self.sampler, List):
#                 limit_len = min(len(ds) for ds in self.sampler)
#             else:
#                 limit_len = min(len(ds) for _, ds in self.sampler.items())
#         else:
#             limit_len = 0
#         # 最后一个批次的大小
#         last_batch_size = limit_len % self.batch_size
#
#         while True:
#             # 全部采样完，退出
#             if len(samplers) == 0:
#                 break
#             batch, flag = [], False
#             # sampler_len代表已经取出来的数据个数
#             sampler, batch_size, index, sampler_len, name = samplers.pop(0)
#             for _ in range(batch_size):
#                 try:
#                     batch.append(index + next(sampler))
#                     sampler_len += 1
#                 except StopIteration:
#                     flag = True
#                     # ds_ratio为None，第一种情况，删除掉采样完的数据即可。
#                     if self.ds_ratio == 'pad_to_most' and sampler_len < limit_len:
#                         # 重置sampler，并取足一个batch数据
#                         sampler = iter(self.sampler[name])
#                         # 由于batch_size一定小于等于ds的长度，故能够取足一个batch_size的数据
#                         for _ in range(batch_size-len(batch)):
#                             batch.append(next(sampler) + index)
#                             sampler_len += 1
#                     break
#
#             # ds_ratio不为None情况
#             # 两种情况会触发一下逻辑：1.truncate_to_least时，最短的数据集最后一个batch大小不等于batch_size时，
#             # 其他较长的数据集的最后一个batch长度会较长；2. pad_to_most，最长的数据集最后一个batch不等于batch_size时，较短数据集最后一个
#             # batch长度会较长
#             if limit_len != 0 and limit_len < sampler_len:
#                 batch = batch[:last_batch_size]
#             # ds_ratio为任意情况下， 没有取完所有数据，则添加到队列尾部
#             elif (limit_len == 0 and flag == False) or limit_len > sampler_len:
#                 samplers.append((sampler, batch_size, index, sampler_len, name))
#             if len(batch) == batch_size:
#                 yield batch
#             elif len(batch) > 0 and not self.drop_last:
#                 yield batch
#
#     def __len__(self) -> int:
#         lens = 0
#         max_len, ds_len = 0, 0
#         if self.ds_ratio == 'truncate_to_least':
#             if isinstance(self.sampler, List):
#                 max_len = min(len(sampler) for sampler in self.sampler)
#                 ds_len = len(self.sampler)
#             elif isinstance(self.sampler, Dict):
#                 max_len = min(len(sampler) for _, sampler in self.sampler.items())
#                 for _, _ in self.sampler.items():
#                     ds_len += 1
#
#         elif self.ds_ratio == 'pad_to_most':
#             if isinstance(self.sampler, List):
#                 max_len = max(len(sampler) for sampler in self.sampler)
#                 ds_len = len(self.sampler)
#             elif isinstance(self.sampler, Dict):
#                 max_len = max(len(sampler) for _, sampler in self.sampler.items())
#                 for _, _ in self.sampler.items():
#                     ds_len += 1
#         if self.ds_ratio is None:
#             if isinstance(self.sampler, List):
#                 for i in range(len(self.sampler)):
#                     sampler = self.sampler[i]
#                     if self.drop_last:
#                         lens += len(sampler) // self.batch_size
#                     else:
#                         lens += (len(sampler) + self.batch_size - 1) // self.batch_size
#             elif isinstance(self.sampler, Dict):
#                 for name, sampler in self.sampler.items():
#                     if self.drop_last:
#                         lens += len(sampler) // self.batch_size
#                     else:
#                         lens += (len(sampler) + self.batch_size - 1) // self.batch_size
#         else:
#             for i in range(ds_len):
#                 if self.drop_last:
#                     lens += max_len // self.batch_size
#                 else:
#                     lens += (max_len + self.batch_size - 1) // self.batch_size
#         return lens


class BucketSampler(Sampler):
    r"""
    带Bucket的 `Random Sampler`. 可以随机地取出长度相似的元素
    """

    def __init__(self, dataset, num_buckets=10, batch_size=None, seq_len_field_name='seq_len', drop_last=False) -> None:
        r"""
        
        :param int num_buckets: bucket的数量
        :param int batch_size: batch的大小. 默认为None，Trainer/Tester在调用BucketSampler时，会将该值正确设置，如果是非
            Trainer/Tester场景使用，需要显示传递该值
        :param str seq_len_field_name: 对应序列长度的 `field` 的名字
        """
        self.dataset = dataset
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.seq_len_field_name = seq_len_field_name

    def set_batch_size(self, batch_size) -> None:
        r"""

        :param int batch_size: 每个batch的大小
        :return:
        """
        self.batch_size = batch_size

    def __iter__(self):
        if self.batch_size is None:
            raise RuntimeError("batch_size is None.")
        seq_lens = self.dataset.get_all_fields()[self.seq_len_field_name].content
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

        return chain(*batchs)


class ConstTokenNumSampler(Sampler):
    """
    尽量保证每个batch的输入token数量是接近的。

    """

    def __init__(self, dataset, seq_len_field_name: List[int], max_token: int = 4096, max_sentence: int = -1,
                 need_be_multiple_of: int = 1, num_bucket: int = -1) -> None:
        """

        :param dataset:
        :param List[int] seq_len_field_name: 哪个field指示的sample的长度
        :param int max_token: 每个batch的最大的token数量
        :param int max_sentence: 每个batch最多多少个instance, -1表示根据max_token决定
        :param int need_be_multiple_of: 生成的batch的instance的数量需要是几的倍数，在DataParallel场景下会用到
        :param int num_bucket: 将数据按长度拆分为num_bucket个bucket，batch中的sample尽量在bucket之中进行组合，这样可以减少padding。
        """
        assert (max_sentence != -1 and max_sentence >= need_be_multiple_of) or max_sentence < 1
        self.dataset = dataset
        self.seq_len_field_name = seq_len_field_name
        self.num_bucket = num_bucket
        self.max_token = max_token
        self._max_sentence = max_sentence
        self.need_be_multiple_of = need_be_multiple_of

        assert len(self.dataset) > self.num_bucket, "The number of samples should be larger than buckets."
        seq_len = self.dataset.get_field(self.seq_len_field_name)
        self.seq_len = seq_len
        seq_len_indice = [(length, i) for i, length in enumerate(seq_len)]
        seq_len_indice.sort(key=lambda x: x[0])
        indice_in_buckets = []
        if self.num_bucket > 0:
            sample_per_bucket = len(seq_len_indice) // self.num_bucket
            i = 0
            while len(indice_in_buckets) < len(seq_len_indice):
                indice_in_buckets.append(seq_len_indice[i * sample_per_bucket:(i + 1) * sample_per_bucket])
                i += 1
        else:
            indice_in_buckets = [seq_len_indice]
        self.indice_in_buckets = indice_in_buckets
        self.get_new_order()

    @property
    def max_sentence(self):
        if self._max_sentence < 1:
            return 100000000
        return self._max_sentence

    @max_sentence.setter
    def max_sentence(self, max_sentence):
        self._max_sentence = max_sentence

    def get_new_order(self) -> None:
        np.random.shuffle(self.indice_in_buckets)
        for bucket in self.indice_in_buckets:
            np.random.shuffle(bucket)
        indices = list(chain(*self.indice_in_buckets))
        batches = []
        cur_max_len = 0
        batch = []
        for length, i in indices:
            max_len = max(length, cur_max_len)
            if max_len * (len(batch) + 1) > self.max_token or len(batch) >= self.max_sentence:
                left_sample = len(batch) % self.need_be_multiple_of
                add_samples = batch.copy()
                cur_max_len = length
                if left_sample != 0:
                    add_samples = add_samples[:-left_sample]
                    batch = batch[-left_sample:]
                    cur_max_len = max(cur_max_len, max(batch))
                else:
                    batch = []
                if len(add_samples) == 0:
                    raise RuntimeError(
                        f"The sample `{i}` is too long to make a batch with {self.need_be_multiple_of} samples.")
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

    def __iter__(self) -> Iterable[int]:
        for batch in self.batches:
            yield batch
        self.get_new_order()

    def __len__(self):
        return len(self.batches)


class ConstantTokenNumSampler:
    """
    尽量保证每个batch的输入token数量是接近的。

    """

    def __init__(self, seq_len, max_token: List[int] = 4096, max_sentence: int = -1,
                 need_be_multiple_of: int = 1, num_bucket: int = -1) -> None:
        """

        :param List[int] seq_len: list[int], 是每个sample的长度。一般可以通过dataset.get_field('seq_len').content传入
        :param int max_token: 每个batch的最大的token数量
        :param int max_sentence: 每个batch最多多少个instance, -1表示根据max_token决定
        :param int need_be_multiple_of: 生成的batch的instance的数量需要是几的倍数，在DataParallel场景下会用到
        :param int num_bucket: 将数据按长度拆分为num_bucket个bucket，batch中的sample尽量在bucket之中进行组合，这样可以减少padding。
        """
        assert (max_sentence != -1 and max_sentence >= need_be_multiple_of) or max_sentence < 1
        assert len(seq_len) > num_bucket, "The number of samples should be larger than buckets."
        self.seq_len = seq_len
        self.max_token = max_token
        self._max_sentence = max_sentence
        self.need_be_multiple_of = need_be_multiple_of
        seq_len_indice = [(length, i) for i, length in enumerate(seq_len)]
        seq_len_indice.sort(key=lambda x: x[0])
        indice_in_buckets = []
        if num_bucket > 0:
            sample_per_bucket = len(seq_len_indice) // num_bucket
            i = 0
            while len(indice_in_buckets) < len(seq_len_indice):
                indice_in_buckets.append(seq_len_indice[i * sample_per_bucket:(i + 1) * sample_per_bucket])
                i += 1
        else:
            indice_in_buckets = [seq_len_indice]
        self.indice_in_buckets = indice_in_buckets
        self.get_new_order()

    @property
    def max_sentence(self):
        if self._max_sentence < 1:
            return 100000000
        return self._max_sentence

    @max_sentence.setter
    def max_sentence(self, max_sentence):
        self._max_sentence = max_sentence

    def get_new_order(self) -> None:
        np.random.shuffle(self.indice_in_buckets)
        for bucket in self.indice_in_buckets:
            np.random.shuffle(bucket)
        indices = list(chain(*self.indice_in_buckets))
        batches = []
        cur_max_len = 0
        batch = []
        for length, i in indices:
            max_len = max(length, cur_max_len)
            if max_len * (len(batch) + 1) > self.max_token or len(batch) >= self.max_sentence:
                left_sample = len(batch) % self.need_be_multiple_of
                add_samples = batch.copy()
                cur_max_len = length
                if left_sample != 0:
                    add_samples = add_samples[:-left_sample]
                    batch = batch[-left_sample:]
                    cur_max_len = max(cur_max_len, max(batch))
                else:
                    batch = []
                if len(add_samples) == 0:
                    raise RuntimeError(
                        f"The sample `{i}` is too long to make a batch with {self.need_be_multiple_of} samples.")
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

    def __iter__(self) -> Iterable[int]:
        for batch in self.batches:
            yield batch
        self.get_new_order()

    def __len__(self):
        return len(self.batches)


class SortedSampler(Sampler):
    r"""
    按照sample的长度进行排序，主要在测试的时候使用，可以加速测试（因为减少了padding）
    """

    def __init__(self, dataset, seq_len_field_name: str = 'seq_len', descending: bool = True) -> None:
        """

        :param str seq_len_field_name: 按哪个field进行排序。如果传入的field是数字，则直接按照该数字大小排序；如果传入的field不是
            数字，则使用该field的长度进行排序
        :param bool descending: 是否降序排列
        """
        self.dataset = dataset
        self.seq_len_field_name = seq_len_field_name
        self.descending = descending

    def __iter__(self) -> Iterable[int]:
        seq_lens = self.dataset.get_field(self.seq_len_field_name).content
        try:
            seq_lens = list(map(len, seq_lens))
        except:
            pass

        orders = np.argsort(seq_lens).tolist()  # 从小到大的顺序
        if self.descending:
            orders = orders[::-1]
        for order in orders:
            yield order


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


class UnrepeatedDistributedSampler:
    def __init__(self, dataset, shuffle: bool = False, seed: int = 0):
        """
        考虑在多卡evaluate的场景下，不能重复sample。

        :param dataset:
        :param shuffle:
        :param seed:
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed

        # 多卡的相关的参数
        self.num_replicas = 1
        self.rank = 0
        self.epoch = -1

    def __len__(self):
        """
        返回 sampler 一次完整的迭代过程会产生多少个index。多卡的情况下，只考虑当前rank；
        :return:
        """
        num_common = len(self.dataset)//self.num_replicas
        self.num_samples = num_common + int(self.rank < (len(self.dataset)-num_common*self.num_replicas))
        return self.num_samples

    def __iter__(self):
        r"""
        当前使用num_consumed_samples做法会在交替使用的时候遇到问题；
        Example:
            >>> sampler = RandomSampler()
            >>> iter1 = iter(sampler)
            >>> iter2 = iter(sampler)
            >>> next(iter1)
            >>> next(iter2)  # 当前num_consumed_samples的数量会发生变化
        """

        indices = self.generate_indices()

        # subsample
        indices = indices[self.rank:len(indices):self.num_replicas]
        assert len(indices) == len(self)

        for index in indices:
            yield index

    def generate_indices(self) -> List[int]:
        """
        生成随机序列

        :return:
        """
        if self.shuffle:
            indices = list(range(len(self.dataset)))
            seed = self.seed + self.epoch
            rng = np.random.default_rng(abs(seed))
            rng.shuffle(indices)
            if self.epoch < 0:  # 防止用户忘记调用 set_epoch，至少这样可以保证每次epoch出来的index顺序不同。
                self.epoch -= 1
        else:
            indices = list(range(len(self.dataset)))
        return indices

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_distributed(self, num_replicas, rank):
        """
        该方法本质上等同于 ddp 情形下的没有完成的初始化，应当在初始化该 sampler 本身后立即被调用；

        :param num_replicas:
        :param rank:
        :return:
        """
        assert num_replicas>0 and isinstance(num_replicas, int)
        assert isinstance(rank, int) and 0<=rank<num_replicas
        # 注意初始化该函数时，所有的状态都应当默认是一个 epoch 刚开始训练的状态；
        self.num_replicas = num_replicas
        self.rank = rank

        return self