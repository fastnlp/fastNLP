class BaseLoader(object):
    """docstring for BaseLoader"""

    def __init__(self, data_name, data_path):
        super(BaseLoader, self).__init__()
        self.data_name = data_name
        self.data_path = data_path

    def load(self):
        """
        :return: string
        """
        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text
