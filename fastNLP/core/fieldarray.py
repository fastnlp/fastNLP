import numpy as np


class FieldArray(object):
    def __init__(self, name, content, padding_val=0, is_target=False, need_tensor=False):
        self.name = name
        self.content = content
        self.padding_val = padding_val
        self.is_target = is_target
        self.need_tensor = need_tensor
        self.dtype = None

    def __repr__(self):
        # TODO
        return '{}: {}'.format(self.name, self.content.__repr__())

    def append(self, val):
        self.content.append(val)

    def __getitem__(self, name):
        return self.get(name)

    def __setitem__(self, name, val):
        assert isinstance(name, int)
        self.content[name] = val

    def get(self, idxes):
        if isinstance(idxes, int):
            return self.content[idxes]
        assert self.need_tensor is True
        batch_size = len(idxes)
        # TODO 当这个fieldArray是seq_length这种只有一位的内容时，不需要padding，需要再讨论一下
        if isinstance(self.content[0], int) or isinstance(self.content[0], float):
            if self.dtype is None:
                self.dtype = np.int64 if isinstance(self.content[0], int) else np.double
            array = np.array([self.content[i] for i in idxes], dtype=self.dtype)
        else:
            if self.dtype is None:
                self.dtype = np.int64
            max_len = max([len(self.content[i]) for i in idxes])
            array = np.full((batch_size, max_len), self.padding_val, dtype=self.dtype)

            for i, idx in enumerate(idxes):
                array[i][:len(self.content[idx])] = self.content[idx]
        return array

    def __len__(self):
        return len(self.content)
