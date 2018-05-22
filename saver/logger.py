from saver.base_saver import BaseSaver


class Logger(BaseSaver):
    """Logging"""

    def __init__(self, save_path):
        super(Logger, self).__init__(save_path)

    def log(self, string):
        raise NotImplementedError
