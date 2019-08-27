"""undocumented"""
import numpy as np

from .pipe import Pipe
from .utils import get_tokenizer, _indexize, _add_words_field, _drop_empty_instance
from ..loader.json import JsonLoader
from ..data_bundle import DataBundle
from ..loader.classification import IMDBLoader, YelpFullLoader, SSTLoader, SST2Loader, YelpPolarityLoader
from ...core.const import Const
from ...core.dataset import DataSet
from ...core.instance import Instance
from ...core.vocabulary import Vocabulary


WORD_PAD = "[PAD]"
WORD_UNK = "[UNK]"
DOMAIN_UNK = "X"
TAG_UNK = "X"



class ExtCNNDMPipe(Pipe):
    def __init__(self, vocab_size, vocab_path, sent_max_len, doc_max_timesteps, domain=False):
        self.vocab_size = vocab_size
        self.vocab_path = vocab_path
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps
        self.domain = domain


    def process(self, db: DataBundle):
        """
        传入的DataSet应该具备如下的结构

        .. csv-table::
           :header: "text", "summary", "label", "domain"

           "I got 'new' tires from them and... ", "The 'new' tires...", [0, 1], "cnndm"
           "Don't waste your time.  We had two...", "Time is precious", [1], "cnndm"
           "...", "...", [], "cnndm"

        :param data_bundle:
        :return:
        """

        db.apply(lambda x: _lower_text(x['text']), new_field_name='text')
        db.apply(lambda x: _lower_text(x['summary']), new_field_name='summary')
        db.apply(lambda x: _split_list(x['text']), new_field_name='text_wd')
        db.apply(lambda x: _split_list(x['summary']), new_field_name='summary_wd')
        db.apply(lambda x: _convert_label(x["label"], len(x["text"])), new_field_name="flatten_label")

        db.apply(lambda x: _pad_sent(x["text_wd"], self.sent_max_len), new_field_name="pad_text_wd")
        db.apply(lambda x: _token_mask(x["text_wd"], self.sent_max_len), new_field_name="pad_token_mask")
        # pad document
        db.apply(lambda x: _pad_doc(x["pad_text_wd"], self.sent_max_len, self.doc_max_timesteps), new_field_name=Const.INPUT)
        db.apply(lambda x: _sent_mask(x["pad_text_wd"], self.doc_max_timesteps), new_field_name=Const.INPUT_LEN)
        db.apply(lambda x: _pad_label(x["flatten_label"], self.doc_max_timesteps), new_field_name=Const.TARGET)

        db = _drop_empty_instance(db, "label")

        # set input and target
        db.set_input(Const.INPUT, Const.INPUT_LEN)
        db.set_target(Const.TARGET, Const.INPUT_LEN)

        print("[INFO] Load existing vocab from %s!" % self.vocab_path)
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
        db.set_vocab(vocabs, "vocab")

        if self.domain == True:
            domaindict = Vocabulary(padding=None, unknown=DOMAIN_UNK)
            domaindict.from_dataset(db, field_name="publication")
            db.set_vocab(domaindict, "domain")

        return db


    def process_from_file(self, paths=None):
        """
            :param paths:
            :return: DataBundle
            """
        db = DataBundle()
        if isinstance(paths, dict):
            for key, value in paths.items():
                db.set_dataset(JsonLoader()._load(value), key)
        else:
            db.set_dataset(JsonLoader()._load(paths), 'test')
        self.process(db)
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


