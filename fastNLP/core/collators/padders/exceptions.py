__all__ = [
    'InconsistencyError',
    'EleDtypeUnsupportedError',
    'EleDtypeDtypeConversionError',
    'DtypeUnsupportedError',
    "DtypeError"
]


class InconsistencyError(BaseException):
    """
    当一个 batch 中的数据存在 shape，dtype 之类的不一致时的报错。

    """
    def __init__(self, msg, *args):
        super(InconsistencyError, self).__init__(msg, *args)


class DtypeError(BaseException):
    def __init__(self, msg, *args):
        super(DtypeError, self).__init__(msg, *args)
        self.msg = msg


class EleDtypeUnsupportedError(DtypeError):
    """
    当 batch 中的 element 的类别本身无法 pad 的时候报错。
    例如要求 str 类型的数据进行 padding 。

    """


class EleDtypeDtypeConversionError(DtypeError):
    """
    当 batch 中的 element 的类别无法转换为 dtype 类型时报错。

    """


class DtypeUnsupportedError(DtypeError):
    """
    当当前 backend 不支持这种类型的 dtype 时报错。

    """