class BaseLoader(object):
    """docstring for BaseLoader"""

    def __init__(self, data_path):
        super(BaseLoader, self).__init__()
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
        return [line.strip() for line in text]


class ToyLoader0(BaseLoader):
    """
        For charLM
    """

    def __init__(self, data_path):
        super(ToyLoader0, self).__init__(data_path)

    def load(self):
        with open(self.data_path, 'r') as f:
            corpus = f.read().lower()
        import re
        corpus = re.sub(r"<unk>", "unk", corpus)
        return corpus.split()
