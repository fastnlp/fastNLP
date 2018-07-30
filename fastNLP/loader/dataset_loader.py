import os

from fastNLP.loader.base_loader import BaseLoader


class DatasetLoader(BaseLoader):
    """"loader for data sets"""

    def __init__(self, data_name, data_path):
        super(DatasetLoader, self).__init__(data_name, data_path)


class POSDatasetLoader(DatasetLoader):
    """Dataset Loader for POS Tag datasets.

    In these datasets, each line are divided by '\t'
    while the first Col is the vocabulary and the second
    Col is the label.
        Different sentence are divided by an empty line.
        e.g:
        Tom label1
        and label2
        Jerry   label1
        .   label3
        (separated by an empty line)
        Hello   label4
        world   label5
        !   label3
        In this file, there are two sentence "Tom and Jerry ."
    and "Hello world !". Each word has its own label from label1
    to label5.
    """
    def __init__(self, data_name, data_path):
        super(POSDatasetLoader, self).__init__(data_name, data_path)

    def load(self):
        assert os.path.exists(self.data_path)
        with open(self.data_path, "r", encoding="utf-8") as f:
            line = f.read()
        return line

    def load_lines(self):
        """
        :return data: three-level list
            [
                [ [word_11, word_12, ...], [label_1, label_1, ...] ],
                [ [word_21, word_22, ...], [label_2, label_1, ...] ],
                ...
            ]
        """
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return self.parse(lines)

    @staticmethod
    def parse(lines):
        data = []
        sentence = []
        for line in lines:
            line = line.strip()
            if len(line) > 1:
                sentence.append(line.split('\t'))
            else:
                words = []
                labels = []
                for tokens in sentence:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                data.append([words, labels])
                sentence = []
        if len(sentence) != 0:
            words = []
            labels = []
            for tokens in sentence:
                words.append(tokens[0])
                labels.append(tokens[1])
            data.append([words, labels])
        return data


class TokenizeDatasetLoader(DatasetLoader):
    """
    Data set loader for tokenization data sets
    """

    def __init__(self, data_name, data_path):
        super(TokenizeDatasetLoader, self).__init__(data_name, data_path)

    def load_pku(self, max_seq_len=64):
        """
        load pku dataset for Chinese word segmentation
        CWS (Chinese Word Segmentation) pku training dataset format:
            1. Each line is a sentence.
            2. Each word in a sentence is separated by space.
        This function convert the pku dataset into three-level lists with labels <BMES>.
            B: beginning of a word
            M: middle of a word
            E: ending of a word
            S: single character

        :param max_seq_len: int, the maximum length of a sequence. If a sequence is longer than it, split it into
                several sequences.
        :return: three-level lists
        """
        assert isinstance(max_seq_len, int) and max_seq_len > 0
        with open(self.data_path, "r", encoding="utf-8") as f:
            sentences = f.readlines()
        data = []
        for sent in sentences:
            words = []
            labels = []
            tokens = sent.strip().split()
            for start in range(len(tokens) // max_seq_len):

            for token in token_seq:
                if len(token) == 1:
                    words.append(token)
                    labels.append("S")
                else:
                    words.append(token[0])
                    labels.append("B")
                    for idx in range(1, len(token) - 1):
                        words.append(token[idx])
                        labels.append("M")
                    words.append(token[-1])
                    labels.append("E")
            data.append([words, labels])
        return data


class ClassDatasetLoader(DatasetLoader):
    """Loader for classification data sets"""

    def __init__(self, data_name, data_path):
        super(ClassDatasetLoader, self).__init__(data_name, data_path)

    def load(self):
        assert os.path.exists(self.data_path)
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return self.parse(lines)

    @staticmethod
    def parse(lines):
        """
        Params
            lines: lines from dataset
        Return
            list(list(list())): the three level of lists are
                words, sentence, and dataset
        """
        dataset = list()
        for line in lines:
            line = line.strip().split()
            label = line[0]
            words = line[1:]
            if len(words) <= 1:
                continue

            sentence = [words, label]
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


class LMDatasetLoader(DatasetLoader):
    def __init__(self, data_name, data_path):
        super(LMDatasetLoader, self).__init__(data_name, data_path)

    def load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError("file {} not found.".format(self.data_path))
        with open(self.data_path, "r", encoding="utf=8") as f:
            text = " ".join(f.readlines())
        return text.strip().split()


if __name__ == "__main__":
    """
    data = POSDatasetLoader("xxx", "../../test/data_for_tests/people.txt").load_lines()
    for example in data:
        for w, l in zip(example[0], example[1]):
            print(w, l)
    """

    ans = TokenizeDatasetLoader("xxx", "/home/zyfeng/Desktop/data/icwb2-data/training/test").load_pku()
    print(ans)
