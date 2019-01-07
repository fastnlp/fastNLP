import os
import sys

import torch

# in order to run fastNLP without installation
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


from fastNLP.api.pipeline import Pipeline
from fastNLP.api.processor import SeqLenProcessor
from fastNLP.core.metrics import SpanFPreRecMetric
from fastNLP.core.trainer import Trainer
from fastNLP.io.config_io import ConfigLoader, ConfigSection
from fastNLP.models.sequence_modeling import AdvSeqLabel
from reproduction.chinese_word_segment.process.cws_processor import VocabIndexerProcessor
from reproduction.pos_tag_model.pos_reader import ZhConllPOSReader
from fastNLP.api.processor import ModelProcessor, Index2WordProcessor

cfgfile = './pos_tag.cfg'
pickle_path = "save"


def train():
    # load config
    train_param = ConfigSection()
    model_param = ConfigSection()
    ConfigLoader().load_config(cfgfile, {"train": train_param, "model": model_param})
    print("config loaded")

    # Data Loader
    dataset = ZhConllPOSReader().load("/home/hyan/train.conllx")
    print(dataset)
    print("dataset transformed")

    dataset.rename_field("tag", "truth")

    vocab_proc = VocabIndexerProcessor("words", new_added_filed_name="word_seq")
    tag_proc = VocabIndexerProcessor("truth")
    seq_len_proc = SeqLenProcessor(field_name="word_seq", new_added_field_name="word_seq_origin_len", is_input=True)

    vocab_proc(dataset)
    tag_proc(dataset)
    seq_len_proc(dataset)

    dataset.set_input("word_seq", "word_seq_origin_len", "truth")
    dataset.set_target("truth", "word_seq_origin_len")

    print("processors defined")

    # dataset.set_is_target(tag_ids=True)
    model_param["vocab_size"] = vocab_proc.get_vocab_size()
    model_param["num_classes"] = tag_proc.get_vocab_size()
    print("vocab_size={}  num_classes={}".format(model_param["vocab_size"], model_param["num_classes"]))

    # define a model
    model = AdvSeqLabel(model_param, id2words=tag_proc.vocab.idx2word)

    # call trainer to train
    trainer = Trainer(dataset, model, loss=None, metrics=SpanFPreRecMetric(tag_proc.vocab, pred="predict",
                                                                           target="truth",
                                                                           seq_lens="word_seq_origin_len"),
                      dev_data=dataset, metric_key="f",
                      use_tqdm=False, use_cuda=True, print_every=20, n_epochs=1, save_path="./save")
    trainer.train()

    # save model & pipeline
    model_proc = ModelProcessor(model, seq_len_field_name="word_seq_origin_len")
    id2tag = Index2WordProcessor(tag_proc.vocab, "predict", "tag")

    pp = Pipeline([vocab_proc, seq_len_proc, model_proc, id2tag])
    save_dict = {"pipeline": pp, "model": model, "tag_vocab": tag_proc.vocab}
    torch.save(save_dict, "model_pp.pkl")
    print("pipeline saved")


def infer():
    pass


if __name__ == "__main__":
    train()
