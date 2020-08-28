r"""undocumented"""

__all__ = [
    "CoReferencePipe"
]

import collections

import numpy as np

from fastNLP.core.vocabulary import Vocabulary
from .pipe import Pipe
from ..data_bundle import DataBundle
from ..loader.coreference import CoReferenceLoader
from ...core.const import Const


class CoReferencePipe(Pipe):
    r"""
    对Coreference resolution问题进行处理，得到文章种类/说话者/字符级信息/序列长度。

    处理完成后数据包含文章类别、speaker信息、句子信息、句子对应的index、char、句子长度、target：

        .. csv-table::
           :header: "words1", "words2","words3","words4","chars","seq_len","target"

           "bc", "[[0,0],[1,1]]","[['I','am'],[]]","[[1,2],[]]","[[[1],[2,3]],[]]","[2,3]","[[[2,3],[6,7]],[[10,12],[20,22]]]"
           "[...]", "[...]","[...]","[...]","[...]","[...]","[...]"

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_chars | target | chars | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   |  True  |  True |   True  |
        |  is_target  |   False   |  True  | False |   True  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def process(self, data_bundle: DataBundle):
        r"""
        对load进来的数据进一步处理原始数据包含：raw_key,raw_speaker,raw_words,raw_clusters
        
        .. csv-table::
           :header: "raw_key", "raw_speaker","raw_words","raw_clusters"

           "bc/cctv/00/cctv_0000_0", "[[Speaker#1, Speaker#1],[]]","[['I','am'],[]]","[[[2,3],[6,7]],[[10,12],[20,22]]]"
           "bc/cctv/00/cctv_0000_1", "[['Speaker#1', 'peaker#1'],[]]","[['He','is'],[]]","[[[2,3],[6,7]],[[10,12],[20,22]]]"
           "[...]", "[...]","[...]","[...]"


        :param data_bundle:
        :return:
        """
        genres = {g: i for i, g in enumerate(["bc", "bn", "mz", "nw", "pt", "tc", "wb"])}
        vocab = Vocabulary().from_dataset(*data_bundle.datasets.values(), field_name= Const.RAW_WORDS(3))
        vocab.build_vocab()
        word2id = vocab.word2idx
        data_bundle.set_vocab(vocab, Const.INPUTS(0))
        if self.config.char_path:
            char_dict = get_char_dict(self.config.char_path)
        else:
            char_set = set()
            for i,w in enumerate(word2id):
                if i < 2:
                    continue
                for c in w:
                    char_set.add(c)

            char_dict = collections.defaultdict(int)
            char_dict.update({c: i for i, c in enumerate(char_set)})

        for name, ds in data_bundle.datasets.items():
            # genre
            ds.apply(lambda x: genres[x[Const.RAW_WORDS(0)][:2]], new_field_name=Const.INPUTS(0))

            # speaker_ids_np
            ds.apply(lambda x: speaker2numpy(x[Const.RAW_WORDS(1)], self.config.max_sentences, is_train=name == 'train'),
                     new_field_name=Const.INPUTS(1))

            # sentences
            ds.rename_field(Const.RAW_WORDS(3),Const.INPUTS(2))

            # doc_np
            ds.apply(lambda x: doc2numpy(x[Const.INPUTS(2)], word2id, char_dict, max(self.config.filter),
                                                    self.config.max_sentences, is_train=name == 'train')[0],
                     new_field_name=Const.INPUTS(3))
            # char_index
            ds.apply(lambda x: doc2numpy(x[Const.INPUTS(2)], word2id, char_dict, max(self.config.filter),
                                                    self.config.max_sentences, is_train=name == 'train')[1],
                     new_field_name=Const.CHAR_INPUT)
            # seq len
            ds.apply(lambda x: doc2numpy(x[Const.INPUTS(2)], word2id, char_dict, max(self.config.filter),
                                                    self.config.max_sentences, is_train=name == 'train')[2],
                     new_field_name=Const.INPUT_LEN)

            # clusters
            ds.rename_field(Const.RAW_WORDS(2), Const.TARGET)

            ds.set_ignore_type(Const.TARGET)
            ds.set_padder(Const.TARGET, None)
            ds.set_input(Const.INPUTS(0), Const.INPUTS(1), Const.INPUTS(2), Const.INPUTS(3), Const.CHAR_INPUT, Const.INPUT_LEN)
            ds.set_target(Const.TARGET)

        return data_bundle

    def process_from_file(self, paths):
        bundle = CoReferenceLoader().load(paths)
        return self.process(bundle)


# helper

def doc2numpy(doc, word2id, chardict, max_filter, max_sentences, is_train):
    docvec, char_index, length, max_len = _doc2vec(doc, word2id, chardict, max_filter, max_sentences, is_train)
    assert max(length) == max_len
    assert char_index.shape[0] == len(length)
    assert char_index.shape[1] == max_len
    doc_np = np.zeros((len(docvec), max_len), int)
    for i in range(len(docvec)):
        for j in range(len(docvec[i])):
            doc_np[i][j] = docvec[i][j]
    return doc_np, char_index, length

def _doc2vec(doc,word2id,char_dict,max_filter,max_sentences,is_train):
    max_len = 0
    max_word_length = 0
    docvex = []
    length = []
    if is_train:
        sent_num = min(max_sentences,len(doc))
    else:
        sent_num = len(doc)

    for i in range(sent_num):
        sent = doc[i]
        length.append(len(sent))
        if (len(sent) > max_len):
            max_len = len(sent)
        sent_vec =[]
        for j,word in enumerate(sent):
            if len(word)>max_word_length:
                max_word_length = len(word)
            if word in word2id:
                sent_vec.append(word2id[word])
            else:
                sent_vec.append(word2id["UNK"])
        docvex.append(sent_vec)

    char_index = np.zeros((sent_num, max_len, max_word_length),dtype=int)
    for i in range(sent_num):
        sent = doc[i]
        for j,word in enumerate(sent):
            char_index[i, j, :len(word)] = [char_dict[c] for c in word]

    return docvex,char_index,length,max_len

def speaker2numpy(speakers_raw,max_sentences,is_train):
    if is_train and len(speakers_raw)> max_sentences:
        speakers_raw = speakers_raw[0:max_sentences]
    speakers = flatten(speakers_raw)
    speaker_dict = {s: i for i, s in enumerate(set(speakers))}
    speaker_ids = np.array([speaker_dict[s] for s in speakers])
    return speaker_ids

# 展平
def flatten(l):
    return [item for sublist in l for item in sublist]

def get_char_dict(path):
    vocab = ["<UNK>"]
    with open(path) as f:
        vocab.extend(c.strip() for c in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})
    return char_dict