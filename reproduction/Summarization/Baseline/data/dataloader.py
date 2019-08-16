import pickle
import numpy as np

from fastNLP.core.vocabulary import Vocabulary
from fastNLP.io.data_bundle import DataBundle
from fastNLP.io.dataset_loader import JsonLoader
from fastNLP.core.const import Const

from tools.logger import *

WORD_PAD = "[PAD]"
WORD_UNK = "[UNK]"
DOMAIN_UNK = "X"
TAG_UNK = "X"


class SummarizationLoader(JsonLoader):
    """
    读取summarization数据集，读取的DataSet包含fields::

        text: list(str)，document
        summary: list(str), summary
        text_wd: list(list(str))，tokenized document
        summary_wd: list(list(str)), tokenized summary
        labels: list(int),
        flatten_label: list(int), 0 or 1, flatten labels
        domain: str, optional
        tag: list(str), optional

    数据来源: CNN_DailyMail Newsroom DUC
    """

    def __init__(self):
        super(SummarizationLoader, self).__init__()

    def _load(self, path):
        ds = super(SummarizationLoader, self)._load(path)

        def _lower_text(text_list):
            return [text.lower() for text in text_list]

        def _split_list(text_list):
            return [text.split() for text in text_list]

        def _convert_label(label, sent_len):
            np_label = np.zeros(sent_len, dtype=int)
            if label != []:
                np_label[np.array(label)] = 1
            return np_label.tolist()

        ds.apply(lambda x: _lower_text(x['text']), new_field_name='text')
        ds.apply(lambda x: _lower_text(x['summary']), new_field_name='summary')
        ds.apply(lambda x:_split_list(x['text']), new_field_name='text_wd')
        ds.apply(lambda x:_split_list(x['summary']), new_field_name='summary_wd')
        ds.apply(lambda x:_convert_label(x["label"], len(x["text"])), new_field_name="flatten_label")

        return ds

    def process(self, paths, vocab_size, vocab_path, sent_max_len, doc_max_timesteps, domain=False, tag=False, load_vocab_file=True):
        """
        :param paths: dict  path for each dataset
        :param vocab_size: int  max_size for vocab
        :param vocab_path: str  vocab path
        :param sent_max_len: int    max token number of the sentence
        :param doc_max_timesteps: int   max sentence number of the document
        :param domain: bool  build vocab for publication, use 'X' for unknown
        :param tag: bool  build vocab for tag, use 'X' for unknown
        :param load_vocab_file: bool  build vocab (False) or load vocab (True)
        :return: DataBundle
            datasets: dict  keys correspond to the paths dict
            vocabs: dict  key: vocab(if "train" in paths), domain(if domain=True), tag(if tag=True)
            embeddings: optional
        """

        def _pad_sent(text_wd):
            pad_text_wd = []
            for sent_wd in text_wd:
                if len(sent_wd) < sent_max_len:
                    pad_num = sent_max_len - len(sent_wd)
                    sent_wd.extend([WORD_PAD] * pad_num)
                else:
                    sent_wd = sent_wd[:sent_max_len]
                pad_text_wd.append(sent_wd)
            return pad_text_wd

        def _token_mask(text_wd):
            token_mask_list = []
            for sent_wd in text_wd:
                token_num = len(sent_wd)
                if token_num < sent_max_len:
                    mask = [1] * token_num + [0] * (sent_max_len - token_num)
                else:
                    mask = [1] * sent_max_len
                token_mask_list.append(mask)
            return token_mask_list

        def _pad_label(label):
            text_len = len(label)
            if text_len < doc_max_timesteps:
                pad_label = label + [0] * (doc_max_timesteps - text_len)
            else:
                pad_label = label[:doc_max_timesteps]
            return pad_label

        def _pad_doc(text_wd):
            text_len = len(text_wd)
            if text_len < doc_max_timesteps:
                padding = [WORD_PAD] * sent_max_len
                pad_text = text_wd + [padding] * (doc_max_timesteps - text_len)
            else:
                pad_text = text_wd[:doc_max_timesteps]
            return pad_text

        def _sent_mask(text_wd):
            text_len = len(text_wd)
            if text_len < doc_max_timesteps:
                sent_mask = [1] * text_len + [0] * (doc_max_timesteps - text_len)
            else:
                sent_mask = [1] * doc_max_timesteps
            return sent_mask


        datasets = {}
        train_ds = None
        for key, value in paths.items():
            ds = self.load(value)
            # pad sent
            ds.apply(lambda x:_pad_sent(x["text_wd"]), new_field_name="pad_text_wd")
            ds.apply(lambda x:_token_mask(x["text_wd"]), new_field_name="pad_token_mask")
            # pad document
            ds.apply(lambda x:_pad_doc(x["pad_text_wd"]), new_field_name="pad_text")
            ds.apply(lambda x:_sent_mask(x["pad_text_wd"]), new_field_name="seq_len")
            ds.apply(lambda x:_pad_label(x["flatten_label"]), new_field_name="pad_label")

            # rename field
            ds.rename_field("pad_text", Const.INPUT)
            ds.rename_field("seq_len", Const.INPUT_LEN)
            ds.rename_field("pad_label", Const.TARGET)

            # set input and target
            ds.set_input(Const.INPUT, Const.INPUT_LEN)
            ds.set_target(Const.TARGET, Const.INPUT_LEN)

            datasets[key] = ds
            if "train" in key:
                train_ds = datasets[key]

        vocab_dict = {}
        if load_vocab_file == False:
            logger.info("[INFO] Build new vocab from training dataset!")
            if train_ds == None:
                raise ValueError("Lack train file to build vocabulary!")

            vocabs = Vocabulary(max_size=vocab_size, padding=WORD_PAD, unknown=WORD_UNK)
            vocabs.from_dataset(train_ds, field_name=["text_wd","summary_wd"])
            vocab_dict["vocab"] = vocabs
        else:
            logger.info("[INFO] Load existing vocab from %s!" % vocab_path)
            word_list = []
            with open(vocab_path, 'r', encoding='utf8') as vocab_f:
                cnt = 2 # pad and unk
                for line in vocab_f:
                    pieces = line.split("\t")
                    word_list.append(pieces[0])
                    cnt += 1
                    if cnt > vocab_size:
                        break
            vocabs = Vocabulary(max_size=vocab_size, padding=WORD_PAD, unknown=WORD_UNK)
            vocabs.add_word_lst(word_list)
            vocabs.build_vocab()
            vocab_dict["vocab"] = vocabs

        if domain == True:
            domaindict = Vocabulary(padding=None, unknown=DOMAIN_UNK)
            domaindict.from_dataset(train_ds, field_name="publication")
            vocab_dict["domain"] = domaindict
        if tag == True:
            tagdict = Vocabulary(padding=None, unknown=TAG_UNK)
            tagdict.from_dataset(train_ds, field_name="tag")
            vocab_dict["tag"] = tagdict

        for ds in datasets.values():
            vocab_dict["vocab"].index_dataset(ds, field_name=Const.INPUT, new_field_name=Const.INPUT)

        return DataBundle(vocabs=vocab_dict, datasets=datasets)



