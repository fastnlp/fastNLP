from saver.base_saver import BaseSaver


class ModelSaver(BaseSaver):
    """Save a models"""

    def __init__(self, save_path):
        super(ModelSaver, self).__init__(save_path)
