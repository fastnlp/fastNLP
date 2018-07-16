class BaseSaver(object):
    """base class for all savers"""

    def __init__(self, save_path):
        self.save_path = save_path
