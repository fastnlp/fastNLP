from loader.base_loader import BaseLoader


class DatasetLoader(BaseLoader):
    """"loader for data sets"""

    def __init__(self, data_name, data_path):
        super(DatasetLoader, self).__init__(data_name, data_path)


class ConllLoader(DatasetLoader):
    """loader for conll format files"""

    def __int__(self, data_name, data_path):
        """
        :param  str data_name: the name of the conll data set
        :param str data_path: the path to the conll data set
        """
        super(ConllLoader, self).__init__(data_name, data_path)
        self.data_set = self.parse(self.load())

    def load(self):
        """
        :return: list lines: all lines in a conll file
        """
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines

    @staticmethod
    def parse(lines):
        """
        :param list lines:a list containing all lines in a conll file.
        :return: a 3D list
        """
        sentences = list()
        tokens = list()
        for line in lines:
            if line[0] == "#":
                # skip the comments
                continue
            if line == "\n":
                sentences.append(tokens)
                tokens = []
                continue
            tokens.append(line.split())
        return sentences
