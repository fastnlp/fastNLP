
class EarlyStopException(BaseException):
    r"""
    用于EarlyStop时从Trainer训练循环中跳出。

    """

    def __init__(self, msg):
        super(EarlyStopException, self).__init__(msg)
        self.msg = msg
