import os

from fastNLP.loader.base_loader import BaseLoader


class DatasetLoader(BaseLoader):
    """"loader for data sets"""

    def __init__(self, data_name, data_path):
        super(DatasetLoader, self).__init__(data_name, data_path)


class POSDatasetLoader(DatasetLoader):
    """loader for pos data sets"""

    def __init__(self, data_name, data_path):
        super(POSDatasetLoader, self).__init__(data_name, data_path)
        #self.data_set = self.load()

    def load(self):
        assert os.path.exists(self.data_path)
        with open(self.data_path, "r", encoding="utf-8") as f:
            line = f.read()
        return line

    def load_lines(self):
        assert os.path.exists(self.data_path)
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines



class ClassificationDatasetLoader(DatasetLoader):
    """loader for classfication data sets"""

    def __init__(self, data_name, data_path):
        super(ClassificationDatasetLoader, data_name).__init__()

    def load(self):
        assert os.path.exists(self.data_path)
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return self.parse(lines)

    @staticmethod
    def parse(lines):
        """
        :param lines: lines from dataset
        :return: list(list(list())): the three level of lists are
                words, sentence, and dataset
        """
        dataset = list()
        for line in lines:
            label = line.split(" ")[0]
            words = line.split(" ")[1:]
            word = list([w for w in words])
            sentence = list([word, label])
            dataset.append(sentence)
        return dataset


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
