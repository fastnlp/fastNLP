import argparse
import os
import pickle
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


def load_tencent_embed(embed_path, word2id):
    hit = 0
    with open(embed_path, "rb") as f:
        embed_dict = pickle.load(f)
    embedding_tensor = torch.randn(len(word2id), 200)
    for key in word2id:
        if key in embed_dict:
            embedding_tensor[word2id[key]] = torch.Tensor(embed_dict[key])
            hit += 1
    print("vocab_size={} hit={} hit/vocab_size={}".format(len(word2id), hit, hit / len(word2id)))
    return embedding_tensor


def train(checkpoint=None):
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
    if checkpoint is None:
        # pre_trained = load_tencent_embed("/home/zyfeng/data/char_tencent_embedding.pkl", vocab_proc.vocab.word2idx)
        pre_trained = None
        model = AdvSeqLabel(model_param, id2words=tag_proc.vocab.idx2word, emb=pre_trained)
        print(model)
    else:
        model = torch.load(checkpoint)

    # call trainer to train
    trainer = Trainer(dataset, model, loss=None, metrics=SpanFPreRecMetric(tag_proc.vocab, pred="predict",
                                                                           target="truth",
                                                                           seq_lens="word_seq_origin_len"),
                      dev_data=dataset, metric_key="f",
                      use_tqdm=True, use_cuda=True, print_every=5, n_epochs=6, save_path="./save")
    trainer.train(load_best_model=True)

    # save model & pipeline
    model_proc = ModelProcessor(model, seq_len_field_name="word_seq_origin_len")
    id2tag = Index2WordProcessor(tag_proc.vocab, "predict", "tag")

    pp = Pipeline([vocab_proc, seq_len_proc, model_proc, id2tag])
    save_dict = {"pipeline": pp, "model": model, "tag_vocab": tag_proc.vocab}
    torch.save(save_dict, "model_pp.pkl")
    print("pipeline saved")

    torch.save(model, "./save/best_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--restart", action="store_true", help="whether to continue training")
    parser.add_argument("-cp", "--checkpoint", type=str, help="checkpoint of the trained model")
    args = parser.parse_args()

    if args.restart is True:
        # 继续训练 python train_pos_tag.py -c -cp ./save/best_model.pkl
        if args.checkpoint is None:
            raise RuntimeError("Please provide the checkpoint. -cp ")
        train(args.checkpoint)
    else:
        # 一次训练 python train_pos_tag.py
        train()
