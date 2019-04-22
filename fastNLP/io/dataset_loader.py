import os
import json

from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.io.base_loader import DataLoaderRegister


def convert_seq_dataset(data):
    """Create an DataSet instance that contains no labels.

    :param data: list of list of strings, [num_examples, \*].
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

    :param data: list of list of strings, [num_examples, \*].
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

    :param data: list of list of strings, [num_examples, \*].
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


def download_from_url(url, path):
    from tqdm import tqdm
    import requests

    """Download file"""
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
    chunk_size = 16 * 1024
    total_size = int(r.headers.get('Content-length', 0))
    with open(path, "wb") as file ,\
        tqdm(total=total_size, unit='B', unit_scale=1, desc=path.split('/')[-1]) as t:
        for chunk in r.iter_content(chunk_size):
            if chunk:
                file.write(chunk)
                t.update(len(chunk))
    return

def uncompress(src, dst):
    import zipfile, gzip, tarfile, os

    def unzip(src, dst):
        with zipfile.ZipFile(src, 'r') as f:
            f.extractall(dst)

    def ungz(src, dst):
        with gzip.open(src, 'rb') as f, open(dst, 'wb') as uf:
            length = 16 * 1024 # 16KB
            buf = f.read(length)
            while buf:
                uf.write(buf)
                buf = f.read(length)

    def untar(src, dst):
        with tarfile.open(src, 'r:gz') as f:
            f.extractall(dst)

    fn, ext = os.path.splitext(src)
    _, ext_2 = os.path.splitext(fn)
    if ext == '.zip':
        unzip(src, dst)
    elif ext == '.gz' and ext_2 != '.tar':
        ungz(src, dst)
    elif (ext == '.gz' and ext_2 == '.tar') or ext_2 == '.tgz':
        untar(src, dst)
    else:
        raise ValueError('unsupported file {}'.format(src))


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


class DummyPOSReader(DataSetLoader):
    """A simple reader for a dummy POS tagging dataset.

    In these datasets, each line are divided by "\\\\t". The first Col is the vocabulary and the second
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
        super(DummyPOSReader, self).__init__()

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


DataLoaderRegister.set_reader(DummyPOSReader, 'read_pos')


class DummyCWSReader(DataSetLoader):
    """Load pku dataset for Chinese word segmentation.
    """
    def __init__(self):
        super(DummyCWSReader, self).__init__()

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


class DummyClassificationReader(DataSetLoader):
    """Loader for a dummy classification data set"""

    def __init__(self):
        super(DummyClassificationReader, self).__init__()

    def load(self, data_path):
        assert os.path.exists(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = self.parse(lines)
        return self.convert(data)

    @staticmethod
    def parse(lines):
        """每行第一个token是标签，其余是字/词；由空格分隔。

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


class DummyLMReader(DataSetLoader):
    """A Dummy Language Model Dataset Reader
    """
    def __init__(self):
        super(DummyLMReader, self).__init__()

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
    """人民日报数据集
    """
    def __init__(self):
        super(PeopleDailyCorpusLoader, self).__init__()
        self.pos = True
        self.ner = True

    def load(self, data_path, pos=True, ner=True):
        """

        :param str data_path: 数据路径
        :param bool pos: 是否使用词性标签
        :param bool ner: 是否使用命名实体标签
        :return: a DataSet object
        """
        self.pos, self.ner = pos, ner
        with open(data_path, "r", encoding="utf-8") as f:
            sents = f.readlines()
        examples = []
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
            example = [sent_words]
            if self.pos is True:
                example.append(sent_pos_tag)
            if self.ner is True:
                example.append(sent_ner)
            examples.append(example)
        return self.convert(examples)

    def convert(self, data):
        data_set = DataSet()
        for item in data:
            sent_words = item[0]
            if self.pos is True and self.ner is True:
                instance = Instance(words=sent_words, pos_tags=item[1], ner=item[2])
            elif self.pos is True:
                instance = Instance(words=sent_words, pos_tags=item[1])
            elif self.ner is True:
                instance = Instance(words=sent_words, ner=item[1])
            else:
                instance = Instance(words=sent_words)
            data_set.append(instance)
        data_set.apply(lambda ins: len(ins["words"]), new_field_name="seq_len")
        return data_set


class ConllLoader:
    def __init__(self, headers, indexs=None):
        self.headers = headers
        if indexs is None:
            self.indexs = list(range(len(self.headers)))
        else:
            if len(indexs) != len(headers):
                raise ValueError
            self.indexs = indexs

    def load(self, path):
        datalist = []
        with open(path, 'r', encoding='utf-8') as f:
            sample = []
            start = next(f)
            if '-DOCSTART-' not in start:
                sample.append(start.split())
            for line in f:
                if line.startswith('\n'):
                    if len(sample):
                        datalist.append(sample)
                    sample = []
                elif line.startswith('#'):
                    continue
                else:
                    sample.append(line.split())
            if len(sample) > 0:
                datalist.append(sample)

        data = [self.get_one(sample) for sample in datalist]
        data = filter(lambda x: x is not None, data)

        ds = DataSet()
        for sample in data:
            ins = Instance()
            for name, idx in zip(self.headers, self.indexs):
                ins.add_field(field_name=name, field=sample[idx])
            ds.append(ins)
        return ds

    def get_one(self, sample):
        sample = list(map(list, zip(*sample)))
        for field in sample:
            if len(field) <= 0:
                return None
        return sample


class Conll2003Loader(ConllLoader):
    """Loader for conll2003 dataset
    
        More information about the given dataset cound be found on 
        https://sites.google.com/site/ermasoftware/getting-started/ne-tagging-conll2003-data 

        Deprecated. Use ConllLoader for all types of conll-format files.
    """
    def __init__(self):
        headers = [
            'tokens', 'pos', 'chunks', 'ner',
        ]
        super(Conll2003Loader, self).__init__(headers=headers)


class SNLIDataSetReader(DataSetLoader):
    """A data set loader for SNLI data set.

    """
    def __init__(self):
        super(SNLIDataSetReader, self).__init__()

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
    """Deprecated. Use ConllLoader for all types of conll-format files."""
    def __init__(self):
        pass

    def load(self, path, cut_long_sent=False):
        """
        返回的DataSet只包含raw_sentence这个field，内容为str。
        假定了输入为conll的格式，以空行隔开两个句子，每行共7列，即
        ::

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
                    sample.append(line.strip().split())
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


class NaiveCWSReader(DataSetLoader):
    """
    这个reader假设了分词数据集为以下形式, 即已经用空格分割好内容了
    例如::

        这是 fastNLP , 一个 非常 good 的 包 .
    
    或者,即每个part后面还有一个pos tag
    例如::

        也/D  在/P  團員/Na  之中/Ng  ，/COMMACATEGORY

    """

    def __init__(self, in_word_splitter=None):
        super(NaiveCWSReader, self).__init__()
        self.in_word_splitter = in_word_splitter

    def load(self, filepath, in_word_splitter=None, cut_long_sent=False):
        """
        允许使用的情况有(默认以\\\\t或空格作为seg)::
        
            这是 fastNLP , 一个 非常 good 的 包 .
            
        和::
        
            也/D  在/P  團員/Na  之中/Ng  ，/COMMACATEGORY
            
        如果splitter不为None则认为是第二种情况, 且我们会按splitter分割"也/D", 然后取第一部分. 例如"也/D".split('/')[0]
        :param filepath:
        :param in_word_splitter:
        :param cut_long_sent:
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
    """读取中文Conll格式。返回“字级别”的标签，使用BMES记号扩展原来的词级别标签。

        Deprecated. Use ConllLoader for all types of conll-format files.
    """
    def __init__(self):
        pass

    def load(self, path):
        """
        返回的DataSet, 包含以下的field::
        
            words：list of str,
            tag: list of str, 被加入了BMES tag, 比如原来的序列为['VP', 'NN', 'NN', ..]，会被认为是["S-VP", "B-NN", "M-NN",..]
            
        假定了输入为conll的格式，以空行隔开两个句子，每行共7列，即::

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


class ConllxDataLoader(ConllLoader):
    """返回“词级别”的标签信息，包括词、词性、（句法）头依赖、（句法）边标签。跟``ZhConllPOSReader``完全不同。

        Deprecated. Use ConllLoader for all types of conll-format files.
    """
    def __init__(self):
        headers = [
            'words', 'pos_tags', 'heads', 'labels',
        ]
        indexs = [
            1, 3, 6, 7,
        ]
        super(ConllxDataLoader, self).__init__(headers=headers, indexs=indexs)


class SSTLoader(DataSetLoader):
    """load SST data in PTB tree format
        data source: https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
    """
    def __init__(self, subtree=False, fine_grained=False):
        self.subtree = subtree

        tag_v = {'0':'very negative', '1':'negative', '2':'neutral',
                 '3':'positive', '4':'very positive'}
        if not fine_grained:
            tag_v['0'] = tag_v['1']
            tag_v['4'] = tag_v['3']
        self.tag_v = tag_v

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
            datas = []
            for l in f:
                datas.extend([(s, self.tag_v[t])
                              for s, t in self.get_one(l, self.subtree)])
        ds = DataSet()
        for words, tag in datas:
            ds.append(Instance(words=words, raw_tag=tag))
        return ds

    @staticmethod
    def get_one(data, subtree):
        from nltk.tree import Tree
        tree = Tree.fromstring(data)
        if subtree:
            return [(t.leaves(), t.label()) for t in tree.subtrees()]
        return [(tree.leaves(), tree.label())]


class JsonLoader(DataSetLoader):
    """Load json-format data,
        every line contains a json obj, like a dict
        fields is the dict key that need to be load
    """
    def __init__(self, **fields):
        super(JsonLoader, self).__init__()
        self.fields = {}
        for k, v in fields.items():
            self.fields[k] = k if v is None else v

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            datas = [json.loads(l) for l in f]
        ds = DataSet()
        for d in datas:
            ins = Instance()
            for k, v in d.items():
                if k in self.fields:
                    ins.add_field(self.fields[k], v)
            ds.append(ins)
        return ds


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

