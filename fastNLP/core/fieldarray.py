import torch
import numpy as np

class FieldArray(object):
    def __init__(self, name, content, padding_val=0, is_target=True, need_tensor=True):
        self.name = name
        self.data = [self._convert_np(val) for val in content]
        self.padding_val = padding_val
        self.is_target = is_target
        self.need_tensor = need_tensor

    def _convert_np(self, val):
        if not isinstance(val, np.array):
            return np.array(val)
        else:
            return val

    def append(self, val):
        self.data.append(self._convert_np(val))

    def get(self, idxes):
        if isinstance(idxes, int):
            return self.data[idxes]
        elif isinstance(idxes, list):
            id_list = np.array(idxes)
        batch_size = len(id_list)
        len_list = [(i, self.data[i].shape[0]) for i in id_list]
        _, max_len = max(len_list, key=lambda x: x[1])
        array = np.full((batch_size, max_len), self.padding_val, dtype=np.int32)

        for i, (idx, length) in enumerate(len_list):
            if length == max_len:
                array[i] = self.data[idx]
            else:
                array[i][:length] = self.data[idx]
        return array

    def __len__(self):
        return len(self.data)
