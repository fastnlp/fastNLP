class BaseLoader(object):

    def __init__(self):
        super(BaseLoader, self).__init__()

    @staticmethod
    def load_lines(data_path):
        with open(data_path, "r", encoding="utf=8") as f:
            text = f.readlines()
        return [line.strip() for line in text]

    @staticmethod
    def load(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.readlines()
        return [[word for word in sent.strip()] for sent in text]


class ToyLoader0(BaseLoader):
    """
        For CharLM
    """

    def __init__(self, data_path):
        super(ToyLoader0, self).__init__(data_path)

    def load(self):
        with open(self.data_path, 'r') as f:
            corpus = f.read().lower()
        import re
        corpus = re.sub(r"<unk>", "unk", corpus)
        return corpus.split()
