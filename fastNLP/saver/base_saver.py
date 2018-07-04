class BaseSaver(object):
    """base class for all savers"""

    def __init__(self, save_path):
        self.save_path = save_path

    def save_bytes(self):
        raise NotImplementedError

    def save_str(self):
        raise NotImplementedError

    def compress(self):
        raise NotImplementedError
