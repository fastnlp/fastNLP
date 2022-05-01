
class Padder:
    def __init__(self, pad_val, dtype):
        self.pad_val = pad_val
        self.dtype = dtype

    def __call__(self, batch_field):
        return self.pad(batch_field=batch_field, pad_val=self.pad_val, dtype=self.dtype)

    @staticmethod
    def pad(batch_field, pad_val, dtype):
        raise NotImplementedError()


class NullPadder(Padder):
    def __init__(self, ele_dtype=None, pad_val=None, dtype=None):
        """
        不进行任何 检查 与 pad 的空 padder 。

        :param ele_dtype:
        :param pad_val:
        :param dtype:
        """
        super().__init__(pad_val=pad_val, dtype=dtype)

    def __call__(self, batch_field):
        # 直接返回，不调用 pad() 方法加快速度。
        return batch_field
