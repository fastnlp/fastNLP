__all__ = [
    're_instantiate_sampler'
]
from array import array
from typing import Sequence
from collections import deque


def re_instantiate_sampler(sampler, new_sampler_class=None):
    all_attributes = vars(sampler)
    if new_sampler_class is not None:
        return new_sampler_class(**all_attributes)
    return type(sampler)(**all_attributes)


def create_array(length, fill_value) -> array:
    """
    根据长度自动创建 array ，超过 4294967295 需要使用 'L', 否则使用 'I'

    :param length:
    :param fill_value:
    :return:
    """
    if not isinstance(fill_value, Sequence):
        fill_value = [fill_value]*length

    if length > 4294967295:
        _index_lst = array("L", fill_value)
    else:
        _index_lst = array("I", fill_value)
    return _index_lst


class NumConsumedSamplesArray:
    def __init__(self, buffer_size=2000, num_consumed_samples=0):
        """
        保留 buffer_size 个 num_consumed_samples 数据，可以索引得到某个 index 下的 num_consumed_samples 多少
        ex:
            array = NumConsumedSamplesArray(buffer_size=3)
            for i in range(10):
                array.push(i)

            array[9]  # 输出为9，表示这个位置真实的 num_consumed_samples 是多少。
            array[6]  # 报错，因为只保留了3个最近的数据，6超过了最大buffer的记录了，即 [7, 8, 9]

        :param buffer_size: 报错多少个历史。
        :param num_consumed_samples: 第一个 num_consumed_samples 是多少。
        """
        self.count = 0
        self.deque = deque(maxlen=buffer_size)
        if num_consumed_samples is not None:
            self.push(num_consumed_samples)
        self.buffer_size = buffer_size

    def __getitem__(self, item):
        if len(self.deque) == 0:  # 如果没有任何缓存的内容，说明还没有写入，直接返回0
            return 0
        assert isinstance(item, int), "Only int index allowed."
        assert self.count-len(self.deque)<=item<self.count, f"Only keep {len(self.deque)} history index."
        index = len(self.deque) - (self.count - item)
        return self.deque[index]

    def push(self, num_consumed_samples):
        self.deque.append(num_consumed_samples)
        self.count += 1