from fastNLP.loader.base_loader import BaseLoader
import os


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
            lines = f.readlines()
        return self.parse(lines)

    @staticmethod
    def parse(lines):
        """
        :param lines: lines from dataset
        :return: list(list(list())): the three level of lists are
                token, sentence, and dataset
        """
        dataset = list()
        for line in lines:
            sentence = list()
            words = line.split(" ")
            for w in words:
                tokens = list()
                tokens.append(w.split('/')[0])
                tokens.append(w.split('/')[1])
                sentence.append(tokens)
            dataset.append(sentence)
        return dataset

class ClassficationDatasetLoader(DatasetLoader):
    """loader for classfication data sets"""

    def __init__(self, data_name, data_path):
        super(ClassficationDatasetLoader, data_name)

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
