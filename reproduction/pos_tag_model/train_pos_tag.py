import torch

from fastNLP.api.pipeline import Pipeline
from fastNLP.api.processor import SeqLenProcessor
from fastNLP.core.metrics import SpanFPreRecMetric
from fastNLP.core.trainer import Trainer
from fastNLP.io.config_io import ConfigLoader, ConfigSection
from fastNLP.models.sequence_modeling import AdvSeqLabel
from reproduction.chinese_word_segment.process.cws_processor import VocabIndexerProcessor
from reproduction.pos_tag_model.pos_reader import ZhConllPOSReader

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

    vocab_proc = VocabIndexerProcessor("words")
    tag_proc = VocabIndexerProcessor("tag")
    seq_len_proc = SeqLenProcessor(field_name="words", new_added_field_name="word_seq_origin_len")

    vocab_proc(dataset)
    tag_proc(dataset)
    seq_len_proc(dataset)

    dataset.rename_field("words", "word_seq")
    dataset.rename_field("tag", "truth")
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
                      use_tqdm=False, use_cuda=True, print_every=20)
    trainer.train()

    # save model & pipeline
    pp = Pipeline([vocab_proc, seq_len_proc])
    save_dict = {"pipeline": pp, "model": model, "tag_vocab": tag_proc.vocab}
    torch.save(save_dict, "model_pp.pkl")
    print("pipeline saved")


def infer():
    pass


if __name__ == "__main__":
    train()
