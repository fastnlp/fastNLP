import os

from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.io.base_loader import DataLoaderRegister


def convert_seq_dataset(data):
    """Create an DataSet instance that contains no labels.

    :param data: list of list of strings, [num_examples, *].
            Example::

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
    """Convert list of data into DataSet.

    :param data: list of list of strings, [num_examples, *].
            Example::

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
    """Convert list of data into DataSet.

    :param data: list of list of strings, [num_examples, *].
            Example::

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


class DataSetLoader:
    """Interface for all DataSetLoaders.

    """

    def load(self, path):
        """Load data from a given file.

        :param str path: file path
        :return: a DataSet object
        """
        raise NotImplementedError

    def convert(self, data):
        """Optional operation to build a DataSet.

        :param data: inner data structure (user-defined) to represent the data.
        :return: a DataSet object
        """
        raise NotImplementedError


class NativeDataSetLoader(DataSetLoader):
    """A simple example of DataSetLoader

    """

    def __init__(self):
        super(NativeDataSetLoader, self).__init__()

    def load(self, path):
        ds = DataSet.read_csv(path, headers=("raw_sentence", "label"), sep="\t")
        ds.set_input("raw_sentence")
        ds.set_target("label")
        return ds


DataLoaderRegister.set_reader(NativeDataSetLoader, 'read_naive')


class RawDataSetLoader(DataSetLoader):
    """A simple example of raw data reader

    """

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


DataLoaderRegister.set_reader(RawDataSetLoader, 'read_rawdata')


class POSDataSetLoader(DataSetLoader):
    """Dataset Loader for a POS Tag dataset.

    In these datasets, each line are divided by "\t". The first Col is the vocabulary and the second
    Col is the label. Different sentence are divided by an empty line.
        E.g::

            Tom label1
            and label2
            Jerry   label1
            .   label3
            (separated by an empty line)
            Hello   label4
            world   label5
            !   label3

    In this example, there are two sentences "Tom and Jerry ." and "Hello world !". Each word has its own label.
    """

    def __init__(self):
        super(POSDataSetLoader, self).__init__()

    def load(self, data_path):
        """
        :return data: three-level list
            Example::
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


DataLoaderRegister.set_reader(POSDataSetLoader, 'read_pos')


class TokenizeDataSetLoader(DataSetLoader):
    """
    Data set loader for tokenization data sets
    """

    def __init__(self):
        super(TokenizeDataSetLoader, self).__init__()

    def load(self, data_path, max_seq_len=32):
        """Load pku dataset for Chinese word segmentation.
        CWS (Chinese Word Segmentation) pku training dataset format:
        1. Each line is a sentence.
        2. Each word in a sentence is separated by space.
        This function convert the pku dataset into three-level lists with labels <BMES>.
        B: beginning of a word
        M: middle of a word
        E: ending of a word
        S: single character

        :param str data_path: path to the data set.
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


class ClassDataSetLoader(DataSetLoader):
    """Loader for a dummy classification data set"""

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

        :param lines: lines from dataset
        :return: list(list(list())): the three level of lists are words, sentence, and dataset
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


class ConllLoader(DataSetLoader):
    """loader for conll format files"""

    def __init__(self):
        super(ConllLoader, self).__init__()

    def load(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = self.parse(lines)
        return self.convert(data)

    @staticmethod
    def parse(lines):
        """
        :param list lines: a list containing all lines in a conll file.
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

    def convert(self, data):
        pass


class LMDataSetLoader(DataSetLoader):
    """Language Model Dataset Loader

    This loader produces data for language model training in a supervised way.
    That means it has X and Y.

    """

    def __init__(self):
        super(LMDataSetLoader, self).__init__()

    def load(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError("file {} not found.".format(data_path))
        with open(data_path, "r", encoding="utf=8") as f:
            text = " ".join(f.readlines())
        tokens = text.strip().split()
        data = self.sentence_cut(tokens)
        return self.convert(data)

    def sentence_cut(self, tokens, sentence_length=15):
        start_idx = 0
        data_set = []
        for idx in range(len(tokens) // sentence_length):
            x = tokens[start_idx * idx: start_idx * idx + sentence_length]
            y = tokens[start_idx * idx + 1: start_idx * idx + sentence_length + 1]
            if start_idx * idx + sentence_length + 1 >= len(tokens):
                # ad hoc
                y.extend(["<unk>"])
            data_set.append([x, y])
        return data_set

    def convert(self, data):
        pass


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


class Conll2003Loader(DataSetLoader):
    """Self-defined loader of conll2003 dataset
    
        More information about the given dataset cound be found on 
        https://sites.google.com/site/ermasoftware/getting-started/ne-tagging-conll2003-data 
    
    """

    def __init__(self):
        super(Conll2003Loader, self).__init__()

    def load(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        ##Parse the dataset line by line
        parsed_data = []
        sentence = []
        tokens = []
        for line in lines:
            if '-DOCSTART- -X- -X- O' in line or line == '\n':
                if sentence != []:
                    parsed_data.append((sentence, tokens))
                    sentence = []
                    tokens = []
                continue

            temp = line.strip().split(" ")
            sentence.append(temp[0])
            tokens.append(temp[1:4])

        return self.convert(parsed_data)

    def convert(self, parsed_data):
        dataset = DataSet()
        for sample in parsed_data:
            label0_list = list(map(
                lambda labels: labels[0], sample[1]))
            label1_list = list(map(
                lambda labels: labels[1], sample[1]))
            label2_list = list(map(
                lambda labels: labels[2], sample[1]))
            dataset.append(Instance(token_list=sample[0],
                                    label0_list=label0_list,
                                    label1_list=label1_list,
                                    label2_list=label2_list))

        return dataset


class SNLIDataSetLoader(DataSetLoader):
    """A data set loader for SNLI data set.

    """

    def __init__(self):
        super(SNLIDataSetLoader, self).__init__()

    def load(self, path_list):
        """

        :param list path_list: A list of file name, in the order of premise file, hypothesis file, and label file.
        :return: A DataSet object.
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
            Example::
                [
                    [ [premise_word_11, premise_word_12, ...], [hypothesis_word_11, hypothesis_word_12, ...], [label_1] ],
                    [ [premise_word_21, premise_word_22, ...], [hypothesis_word_21, hypothesis_word_22, ...], [label_2] ],
                    ...
                ]

        :return: A DataSet object.
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


class ConllCWSReader(object):
    def __init__(self):
        pass

    def load(self, path, cut_long_sent=False):
        """
        返回的DataSet只包含raw_sentence这个field，内容为str。
        假定了输入为conll的格式，以空行隔开两个句子，每行共7列，即
            1	编者按	编者按	NN	O	11	nmod:topic
            2	：	：	PU	O	11	punct
            3	7月	7月	NT	DATE	4	compound:nn
            4	12日	12日	NT	DATE	11	nmod:tmod
            5	，	，	PU	O	11	punct

            1	这	这	DT	O	3	det
            2	款	款	M	O	1	mark:clf
            3	飞行	飞行	NN	O	8	nsubj
            4	从	从	P	O	5	case
            5	外型	外型	NN	O	8	nmod:prep
        """
        datalist = []
        with open(path, 'r', encoding='utf-8') as f:
            sample = []
            for line in f:
                if line.startswith('\n'):
                    datalist.append(sample)
                    sample = []
                elif line.startswith('#'):
                    continue
                else:
                    sample.append(line.split('\t'))
            if len(sample) > 0:
                datalist.append(sample)

        ds = DataSet()
        for sample in datalist:
            # print(sample)
            res = self.get_char_lst(sample)
            if res is None:
                continue
            line = ' '.join(res)
            if cut_long_sent:
                sents = cut_long_sentence(line)
            else:
                sents = [line]
            for raw_sentence in sents:
                ds.append(Instance(raw_sentence=raw_sentence))

        return ds

    def get_char_lst(self, sample):
        if len(sample) == 0:
            return None
        text = []
        for w in sample:
            t1, t2, t3, t4 = w[1], w[3], w[6], w[7]
            if t3 == '_':
                return None
            text.append(t1)
        return text


class POSCWSReader(DataSetLoader):
    """
    支持读取以下的情况, 即每一行是一个词, 用空行作为两句话的界限.
        迈 N
        向 N
        充 N
        ...
        泽 I-PER
        民 I-PER

        （ N
        一 N
        九 N
        ...


    :param filepath:
    :return:
    """

    def __init__(self, in_word_splitter=None):
        super().__init__()
        self.in_word_splitter = in_word_splitter

    def load(self, filepath, in_word_splitter=None, cut_long_sent=False):
        if in_word_splitter is None:
            in_word_splitter = self.in_word_splitter
        dataset = DataSet()
        with open(filepath, 'r') as f:
            words = []
            for line in f:
                line = line.strip()
                if len(line) == 0:  # new line
                    if len(words) == 0:  # 不能接受空行
                        continue
                    line = ' '.join(words)
                    if cut_long_sent:
                        sents = cut_long_sentence(line)
                    else:
                        sents = [line]
                    for sent in sents:
                        instance = Instance(raw_sentence=sent)
                        dataset.append(instance)
                    words = []
                else:
                    line = line.split()[0]
                    if in_word_splitter is None:
                        words.append(line)
                    else:
                        words.append(line.split(in_word_splitter)[0])
        return dataset


class NaiveCWSReader(DataSetLoader):
    """
    这个reader假设了分词数据集为以下形式, 即已经用空格分割好内容了
        这是 fastNLP , 一个 非常 good 的 包 .
    或者,即每个part后面还有一个pos tag
        也/D  在/P  團員/Na  之中/Ng  ，/COMMACATEGORY
    """

    def __init__(self, in_word_splitter=None):
        super().__init__()

        self.in_word_splitter = in_word_splitter

    def load(self, filepath, in_word_splitter=None, cut_long_sent=False):
        """
        允许使用的情况有(默认以\t或空格作为seg)
            这是 fastNLP , 一个 非常 good 的 包 .
        和
            也/D  在/P  團員/Na  之中/Ng  ，/COMMACATEGORY
        如果splitter不为None则认为是第二种情况, 且我们会按splitter分割"也/D", 然后取第一部分. 例如"也/D".split('/')[0]
        :param filepath:
        :param in_word_splitter:
        :return:
        """
        if in_word_splitter == None:
            in_word_splitter = self.in_word_splitter
        dataset = DataSet()
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line.replace(' ', '')) == 0:  # 不能接受空行
                    continue

                if not in_word_splitter is None:
                    words = []
                    for part in line.split():
                        word = part.split(in_word_splitter)[0]
                        words.append(word)
                        line = ' '.join(words)
                if cut_long_sent:
                    sents = cut_long_sentence(line)
                else:
                    sents = [line]
                for sent in sents:
                    instance = Instance(raw_sentence=sent)
                    dataset.append(instance)

        return dataset


def cut_long_sentence(sent, max_sample_length=200):
    """
    将长于max_sample_length的sentence截成多段，只会在有空格的地方发生截断。所以截取的句子可能长于或者短于max_sample_length

    :param sent: str.
    :param max_sample_length: int.
    :return: list of str.
    """
    sent_no_space = sent.replace(' ', '')
    cutted_sentence = []
    if len(sent_no_space) > max_sample_length:
        parts = sent.strip().split()
        new_line = ''
        length = 0
        for part in parts:
            length += len(part)
            new_line += part + ' '
            if length > max_sample_length:
                new_line = new_line[:-1]
                cutted_sentence.append(new_line)
                length = 0
                new_line = ''
        if new_line != '':
            cutted_sentence.append(new_line[:-1])
    else:
        cutted_sentence.append(sent)
    return cutted_sentence


class ZhConllPOSReader(object):
    # 中文colln格式reader
    def __init__(self):
        pass

    def load(self, path):
        """
        返回的DataSet, 包含以下的field
            words：list of str,
            tag: list of str, 被加入了BMES tag, 比如原来的序列为['VP', 'NN', 'NN', ..]，会被认为是["S-VP", "B-NN", "M-NN",..]
        假定了输入为conll的格式，以空行隔开两个句子，每行共7列，即
            1	编者按	编者按	NN	O	11	nmod:topic
            2	：	：	PU	O	11	punct
            3	7月	7月	NT	DATE	4	compound:nn
            4	12日	12日	NT	DATE	11	nmod:tmod
            5	，	，	PU	O	11	punct

            1	这	这	DT	O	3	det
            2	款	款	M	O	1	mark:clf
            3	飞行	飞行	NN	O	8	nsubj
            4	从	从	P	O	5	case
            5	外型	外型	NN	O	8	nmod:prep
        """
        datalist = []
        with open(path, 'r', encoding='utf-8') as f:
            sample = []
            for line in f:
                if line.startswith('\n'):
                    datalist.append(sample)
                    sample = []
                elif line.startswith('#'):
                    continue
                else:
                    sample.append(line.split('\t'))
            if len(sample) > 0:
                datalist.append(sample)

        ds = DataSet()
        for sample in datalist:
            # print(sample)
            res = self.get_one(sample)
            if res is None:
                continue
            char_seq = []
            pos_seq = []
            for word, tag in zip(res[0], res[1]):
                char_seq.extend(list(word))
                if len(word) == 1:
                    pos_seq.append('S-{}'.format(tag))
                elif len(word) > 1:
                    pos_seq.append('B-{}'.format(tag))
                    for _ in range(len(word) - 2):
                        pos_seq.append('M-{}'.format(tag))
                    pos_seq.append('E-{}'.format(tag))
                else:
                    raise ValueError("Zero length of word detected.")

            ds.append(Instance(words=char_seq,
                               tag=pos_seq))

        return ds

    def get_one(self, sample):
        if len(sample) == 0:
            return None
        text = []
        pos_tags = []
        for w in sample:
            t1, t2, t3, t4 = w[1], w[3], w[6], w[7]
            if t3 == '_':
                return None
            text.append(t1)
            pos_tags.append(t2)
        return text, pos_tags


class ConllPOSReader(object):
    # 返回的Dataset包含words(list of list, 里层的list是character), tag两个field(list of str, str是标有BIO的tag)。
    def __init__(self):
        pass

    def load(self, path):
        datalist = []
        with open(path, 'r', encoding='utf-8') as f:
            sample = []
            for line in f:
                if line.startswith('\n'):
                    datalist.append(sample)
                    sample = []
                elif line.startswith('#'):
                    continue
                else:
                    sample.append(line.split('\t'))
            if len(sample) > 0:
                datalist.append(sample)

        ds = DataSet()
        for sample in datalist:
            # print(sample)
            res = self.get_one(sample)
            if res is None:
                continue
            char_seq = []
            pos_seq = []
            for word, tag in zip(res[0], res[1]):
                if len(word) == 1:
                    char_seq.append(word)
                    pos_seq.append('S-{}'.format(tag))
                elif len(word) > 1:
                    pos_seq.append('B-{}'.format(tag))
                    for _ in range(len(word) - 2):
                        pos_seq.append('M-{}'.format(tag))
                    pos_seq.append('E-{}'.format(tag))
                    char_seq.extend(list(word))
                else:
                    raise ValueError("Zero length of word detected.")

            ds.append(Instance(words=char_seq,
                               tag=pos_seq))
        return ds

    def get_one(self, sample):
        if len(sample) == 0:
            return None
        text = []
        pos_tags = []
        for w in sample:
            t1, t2, t3, t4 = w[1], w[3], w[6], w[7]
            if t3 == '_':
                return None
            text.append(t1)
            pos_tags.append(t2)
        return text, pos_tags



class ConllxDataLoader(object):
    def load(self, path):
        """

        :param path: str，存储数据的路径
        :return: DataSet。内含field有'words', 'pos_tags', 'heads', 'labels'(parser的label)
            类似于拥有以下结构, 一行为一个instance(sample)
            words           pos_tags        heads       labels
            ['some', ..]    ['NN', ...]     [2, 3...]   ['nn', 'nn'...]
        """
        datalist = []
        with open(path, 'r', encoding='utf-8') as f:
            sample = []
            for line in f:
                if line.startswith('\n'):
                    datalist.append(sample)
                    sample = []
                elif line.startswith('#'):
                    continue
                else:
                    sample.append(line.split('\t'))
            if len(sample) > 0:
                datalist.append(sample)

        data = [self.get_one(sample) for sample in datalist]
        data_list = list(filter(lambda x: x is not None, data))

        ds = DataSet()
        for example in data_list:
            ds.append(Instance(words=example[0],
                               pos_tags=example[1],
                               heads=example[2],
                               labels=example[3]))
        return ds

    def get_one(self, sample):
        sample = list(map(list, zip(*sample)))
        if len(sample) == 0:
            return None
        for w in sample[7]:
            if w == '_':
                print('Error Sample {}'.format(sample))
                return None
        # return word_seq, pos_seq, head_seq, head_tag_seq
        return sample[1], sample[3], list(map(int, sample[6])), sample[7]


def add_seg_tag(data):
    """

    :param data: list of ([word], [pos], [heads], [head_tags])
    :return: list of ([word], [pos])
    """

    _processed = []
    for word_list, pos_list, _, _ in data:
        new_sample = []
        for word, pos in zip(word_list, pos_list):
            if len(word) == 1:
                new_sample.append((word, 'S-' + pos))
            else:
                new_sample.append((word[0], 'B-' + pos))
                for c in word[1:-1]:
                    new_sample.append((c, 'M-' + pos))
                new_sample.append((word[-1], 'E-' + pos))
        _processed.append(list(map(list, zip(*new_sample))))
    return _processed
