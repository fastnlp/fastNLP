import torch
import numpy as np

class FieldArray(object):
    def __init__(self, name, content, padding_val=0, is_target=True, need_tensor=True):
        self.name = name
        self.content = content
        self.padding_val = padding_val
        self.is_target = is_target
        self.need_tensor = need_tensor

    def append(self, val):
        self.content.append(val)

    def get(self, idxes):
        if isinstance(idxes, int):
            return self.content[idxes]
        batch_size = len(idxes)
        max_len = max([len(self.content[i]) for i in idxes])
        array = np.full((batch_size, max_len), self.padding_val, dtype=np.int32)

        for i, idx in enumerate(idxes):
            array[i][:len(self.content[idx])] = self.content[idx]
        return array

    def __len__(self):
        return len(self.content)
