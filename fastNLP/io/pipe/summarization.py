r"""undocumented"""
import os
import numpy as np

from .pipe import Pipe
from .utils import _drop_empty_instance
from ..loader.summarization import ExtCNNDMLoader
from ..data_bundle import DataBundle
from ...core.const import Const
from ...core.vocabulary import Vocabulary
from ...core._logger import logger


WORD_PAD = "[PAD]"
WORD_UNK = "[UNK]"
DOMAIN_UNK = "X"
TAG_UNK = "X"


class ExtCNNDMPipe(Pipe):
    r"""
    对CNN/Daily Mail数据进行适用于extractive summarization task的预处理，预处理之后的数据，具备以下结构：
    
    .. csv-table::
       :header: "text", "summary", "label", "publication", "text_wd", "words", "seq_len", "target"
    
    """
    def __init__(self, vocab_size, sent_max_len, doc_max_timesteps, vocab_path=None, domain=False):
        r"""
        
        :param vocab_size: int, 词表大小
        :param sent_max_len: int, 句子最大长度，不足的句子将padding，超出的将截断
        :param doc_max_timesteps: int, 文章最多句子个数，不足的将padding，超出的将截断
        :param vocab_path: str, 外部词表路径
        :param domain:  bool, 是否需要建立domain词表
        """
        self.vocab_size = vocab_size
        self.vocab_path = vocab_path
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps
        self.domain = domain

    def process(self, data_bundle: DataBundle):
        r"""
        传入的DataSet应该具备如下的结构

        .. csv-table::
           :header: "text", "summary", "label", "publication"

           ["I got new tires from them and... ","..."], ["The new tires...","..."], [0, 1], "cnndm"
           ["Don't waste your time.  We had two...","..."], ["Time is precious","..."], [1], "cnndm"
           ["..."], ["..."], [], "cnndm"

        :param data_bundle:
        :return: 处理得到的数据包括
         .. csv-table::
           :header: "text_wd", "words", "seq_len", "target"

           [["I","got",..."."],...,["..."]], [[54,89,...,5],...,[9,43,..,0]], [1,1,...,0], [0,1,...,0]
           [["Don't","waste",...,"."],...,["..."]], [[5234,653,...,5],...,[87,234,..,0]], [1,1,...,0], [1,1,...,0]
           [[""],...,[""]], [[],...,[]], [], []
        """

        if self.vocab_path is None:
            error_msg = 'vocab file is not defined!'
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        data_bundle.apply(lambda x: _lower_text(x['text']), new_field_name='text')
        data_bundle.apply(lambda x: _lower_text(x['summary']), new_field_name='summary')
        data_bundle.apply(lambda x: _split_list(x['text']), new_field_name='text_wd')
        data_bundle.apply(lambda x: _convert_label(x["label"], len(x["text"])), new_field_name=Const.TARGET)

        data_bundle.apply(lambda x: _pad_sent(x["text_wd"], self.sent_max_len), new_field_name=Const.INPUT)
        # db.apply(lambda x: _token_mask(x["text_wd"], self.sent_max_len), new_field_name="pad_token_mask")

        # pad document
        data_bundle.apply(lambda x: _pad_doc(x[Const.INPUT], self.sent_max_len, self.doc_max_timesteps), new_field_name=Const.INPUT)
        data_bundle.apply(lambda x: _sent_mask(x[Const.INPUT], self.doc_max_timesteps), new_field_name=Const.INPUT_LEN)
        data_bundle.apply(lambda x: _pad_label(x[Const.TARGET], self.doc_max_timesteps), new_field_name=Const.TARGET)

        data_bundle = _drop_empty_instance(data_bundle, "label")

        # set input and target
        data_bundle.set_input(Const.INPUT, Const.INPUT_LEN)
        data_bundle.set_target(Const.TARGET, Const.INPUT_LEN)

        # print("[INFO] Load existing vocab from %s!" % self.vocab_path)
        word_list = []
        with open(self.vocab_path, 'r', encoding='utf8') as vocab_f:
            cnt = 2  # pad and unk
            for line in vocab_f:
                pieces = line.split("\t")
                word_list.append(pieces[0])
                cnt += 1
                if cnt > self.vocab_size:
                    break
        vocabs = Vocabulary(max_size=self.vocab_size, padding=WORD_PAD, unknown=WORD_UNK)
        vocabs.add_word_lst(word_list)
        vocabs.build_vocab()
        data_bundle.set_vocab(vocabs, "vocab")

        if self.domain is True:
            domaindict = Vocabulary(padding=None, unknown=DOMAIN_UNK)
            domaindict.from_dataset(data_bundle.get_dataset("train"), field_name="publication")
            data_bundle.set_vocab(domaindict, "domain")

        return data_bundle

    def process_from_file(self, paths=None):
        r"""
        :param paths: dict or string
        :return: DataBundle
        """
        loader = ExtCNNDMLoader()
        if self.vocab_path is None:
            if paths is None:
                paths = loader.download()
            if not os.path.isdir(paths):
                error_msg = 'vocab file is not defined!'
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            self.vocab_path = os.path.join(paths, 'vocab')
        db = loader.load(paths=paths)
        db = self.process(db)
        for ds in db.datasets.values():
            db.get_vocab("vocab").index_dataset(ds, field_name=Const.INPUT, new_field_name=Const.INPUT)

        return db


def _lower_text(text_list):
    return [text.lower() for text in text_list]


def _split_list(text_list):
    return [text.split() for text in text_list]


def _convert_label(label, sent_len):
    np_label = np.zeros(sent_len, dtype=int)
    if label != []:
        np_label[np.array(label)] = 1
    return np_label.tolist()


def _pad_sent(text_wd, sent_max_len):
    pad_text_wd = []
    for sent_wd in text_wd:
        if len(sent_wd) < sent_max_len:
            pad_num = sent_max_len - len(sent_wd)
            sent_wd.extend([WORD_PAD] * pad_num)
        else:
            sent_wd = sent_wd[:sent_max_len]
        pad_text_wd.append(sent_wd)
    return pad_text_wd


def _token_mask(text_wd, sent_max_len):
    token_mask_list = []
    for sent_wd in text_wd:
        token_num = len(sent_wd)
        if token_num < sent_max_len:
            mask = [1] * token_num + [0] * (sent_max_len - token_num)
        else:
            mask = [1] * sent_max_len
        token_mask_list.append(mask)
    return token_mask_list


def _pad_label(label, doc_max_timesteps):
    text_len = len(label)
    if text_len < doc_max_timesteps:
        pad_label = label + [0] * (doc_max_timesteps - text_len)
    else:
        pad_label = label[:doc_max_timesteps]
    return pad_label


def _pad_doc(text_wd, sent_max_len, doc_max_timesteps):
    text_len = len(text_wd)
    if text_len < doc_max_timesteps:
        padding = [WORD_PAD] * sent_max_len
        pad_text = text_wd + [padding] * (doc_max_timesteps - text_len)
    else:
        pad_text = text_wd[:doc_max_timesteps]
    return pad_text


def _sent_mask(text_wd, doc_max_timesteps):
    text_len = len(text_wd)
    if text_len < doc_max_timesteps:
        sent_mask = [1] * text_len + [0] * (doc_max_timesteps - text_len)
    else:
        sent_mask = [1] * doc_max_timesteps
    return sent_mask


