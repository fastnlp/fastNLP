import os

from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.io.base_loader import BaseLoader


def convert_seq_dataset(data):
    """Create an DataSet instance that contains no labels.

    :param data: list of list of strings, [num_examples, *].
            ::
            [
                [word_11, word_12, ...],
                ...
            ]

    :return: a DataSet.
    """
    dataset = DataSet()
    for word_seq in data:
        dataset.append(Instance(word_seq=word_seq))
    return dataset


def convert_seq2tag_dataset(data):
    """Convert list of data into DataSet

    :param data: list of list of strings, [num_examples, *].
            ::
            [
                [ [word_11, word_12, ...], label_1 ],
                [ [word_21, word_22, ...], label_2 ],
                ...
            ]

    :return: a DataSet.
    """
    dataset = DataSet()
    for sample in data:
        dataset.append(Instance(word_seq=sample[0], label=sample[1]))
    return dataset


def convert_seq2seq_dataset(data):
    """Convert list of data into DataSet

    :param data: list of list of strings, [num_examples, *].
            ::
            [
                [ [word_11, word_12, ...], [label_1, label_1, ...] ],
                [ [word_21, word_22, ...], [label_2, label_1, ...] ],
                ...
            ]

    :return: a DataSet.
    """
    dataset = DataSet()
    for sample in data:
        dataset.append(Instance(word_seq=sample[0], label_seq=sample[1]))
    return dataset


class DataSetLoader(BaseLoader):
    """"loader for data sets"""

    def __init__(self):
        super(DataSetLoader, self).__init__()

    def load(self, path):
        """ load data in `path` into a dataset
        """
        raise NotImplementedError

    def convert(self, data):
        """convert list of data into dataset
        """
        raise NotImplementedError


@DataSet.set_reader('read_raw')
class RawDataSetLoader(DataSetLoader):
    def __init__(self):
        super(RawDataSetLoader, self).__init__()

    def load(self, data_path, split=None):
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = lines if split is None else [l.split(split) for l in lines]
        lines = list(filter(lambda x: len(x) > 0, lines))
        return self.convert(lines)

    def convert(self, data):
        return convert_seq_dataset(data)


@DataSet.set_reader('read_pos')
class POSDataSetLoader(DataSetLoader):
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

    def __init__(self):
        super(POSDataSetLoader, self).__init__()

    def load(self, data_path):
        """
        :return data: three-level list
            [
                [ [word_11, word_12, ...], [label_1, label_1, ...] ],
                [ [word_21, word_22, ...], [label_2, label_1, ...] ],
                ...
            ]
        """
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = self.parse(lines)
        return self.convert(data)

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

    def convert(self, data):
        """Convert lists of strings into Instances with Fields.
        """
        return convert_seq2seq_dataset(data)


@DataSet.set_reader('read_tokenize')
class TokenizeDataSetLoader(DataSetLoader):
    """
    Data set loader for tokenization data sets
    """

    def __init__(self):
        super(TokenizeDataSetLoader, self).__init__()

    def load(self, data_path, max_seq_len=32):
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
        with open(data_path, "r", encoding="utf-8") as f:
            sentences = f.readlines()
        data = []
        for sent in sentences:
            tokens = sent.strip().split()
            words = []
            labels = []
            for token in tokens:
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
            num_samples = len(words) // max_seq_len
            if len(words) % max_seq_len != 0:
                num_samples += 1
            for sample_idx in range(num_samples):
                start = sample_idx * max_seq_len
                end = (sample_idx + 1) * max_seq_len
                seq_words = words[start:end]
                seq_labels = labels[start:end]
                data.append([seq_words, seq_labels])
        return self.convert(data)

    def convert(self, data):
        return convert_seq2seq_dataset(data)


@DataSet.set_reader('read_class')
class ClassDataSetLoader(DataSetLoader):
    """Loader for classification data sets"""

    def __init__(self):
        super(ClassDataSetLoader, self).__init__()

    def load(self, data_path):
        assert os.path.exists(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = self.parse(lines)
        return self.convert(data)

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

    def convert(self, data):
        return convert_seq2tag_dataset(data)


@DataSet.set_reader('read_people_daily')
class PeopleDailyCorpusLoader(DataSetLoader):
    """
        People Daily Corpus: Chinese word segmentation, POS tag, NER
    """

    def __init__(self):
        super(PeopleDailyCorpusLoader, self).__init__()

    def load(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            sents = f.readlines()

        pos_tag_examples = []
        ner_examples = []
        for sent in sents:
            if len(sent) <= 2:
                continue
            inside_ne = False
            sent_pos_tag = []
            sent_words = []
            sent_ner = []
            words = sent.strip().split()[1:]
            for word in words:
                if "[" in word and "]" in word:
                    ner_tag = "U"
                    print(word)
                elif "[" in word:
                    inside_ne = True
                    ner_tag = "B"
                    word = word[1:]
                elif "]" in word:
                    ner_tag = "L"
                    word = word[:word.index("]")]
                    if inside_ne is True:
                        inside_ne = False
                    else:
                        raise RuntimeError("only ] appears!")
                else:
                    if inside_ne is True:
                        ner_tag = "I"
                    else:
                        ner_tag = "O"
                tmp = word.split("/")
                token, pos = tmp[0], tmp[1]
                sent_ner.append(ner_tag)
                sent_pos_tag.append(pos)
                sent_words.append(token)
            pos_tag_examples.append([sent_words, sent_pos_tag])
            ner_examples.append([sent_words, sent_ner])
        # List[List[List[str], List[str]]]
        # ner_examples not used
        return self.convert(pos_tag_examples)

    def convert(self, data):
        data_set = DataSet()
        for item in data:
            sent_words, sent_pos_tag = item[0], item[1]
            data_set.append(Instance(words=sent_words, tags=sent_pos_tag))
        data_set.apply(lambda ins: len(ins), new_field_name="seq_len")
        data_set.set_target("tags")
        data_set.set_input("sent_words")
        data_set.set_input("seq_len")
        return data_set


class SNLIDataSetLoader(DataSetLoader):
    """A data set loader for SNLI data set.

    """

    def __init__(self):
        super(SNLIDataSetLoader, self).__init__()

    def load(self, path_list):
        """

        :param path_list: A list of file name, in the order of premise file, hypothesis file, and label file.
        :return: data_set: A DataSet object.
        """
        assert len(path_list) == 3
        line_set = []
        for file in path_list:
            if not os.path.exists(file):
                raise FileNotFoundError("file {} NOT found".format(file))

            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                line_set.append(lines)

        premise_lines, hypothesis_lines, label_lines = line_set
        assert len(premise_lines) == len(hypothesis_lines) and len(premise_lines) == len(label_lines)

        data_set = []
        for premise, hypothesis, label in zip(premise_lines, hypothesis_lines, label_lines):
            p = premise.strip().split()
            h = hypothesis.strip().split()
            l = label.strip()
            data_set.append([p, h, l])

        return self.convert(data_set)

    def convert(self, data):
        """Convert a 3D list to a DataSet object.

        :param data: A 3D tensor.
            [
                [ [premise_word_11, premise_word_12, ...], [hypothesis_word_11, hypothesis_word_12, ...], [label_1] ],
                [ [premise_word_21, premise_word_22, ...], [hypothesis_word_21, hypothesis_word_22, ...], [label_2] ],
                ...
            ]
        :return: data_set: A DataSet object.
        """

        data_set = DataSet()

        for example in data:
            p, h, l = example
            # list, list, str
            instance = Instance()
            instance.add_field("premise", p)
            instance.add_field("hypothesis", h)
            instance.add_field("truth", l)
            data_set.append(instance)
        data_set.apply(lambda ins: len(ins["premise"]), new_field_name="premise_len")
        data_set.apply(lambda ins: len(ins["hypothesis"]), new_field_name="hypothesis_len")
        data_set.set_input("premise", "hypothesis", "premise_len", "hypothesis_len")
        data_set.set_target("truth")
        return data_set
