
class EarlyStopException(BaseException):
    r"""
    用于 EarlyStop 时从 Trainer 训练循环中跳出。

    """

    def __init__(self, msg):
        super(EarlyStopException, self).__init__(msg)
        self.msg = msg
