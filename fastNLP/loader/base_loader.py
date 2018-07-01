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

    def load_lines(self):
        with open(self.data_path, "r", encoding="utf=8") as f:
            text = f.readlines()
        return text


class ToyLoader0(BaseLoader):
    """
        For charLM
    """

    def __init__(self, name, path):
        super(ToyLoader0, self).__init__(name, path)

    def load(self):
        with open(self.data_path, 'r') as f:
            corpus = f.read().lower()
        import re
        corpus = re.sub(r"<unk>", "unk", corpus)
        return corpus.split()
