import argparse
import os
import pickle
import sys

import torch

# in order to run fastNLP without installation
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fastNLP.api.pipeline import Pipeline
from fastNLP.api.processor import SeqLenProcessor, VocabIndexerProcessor, SetInputProcessor, IndexerProcessor
from fastNLP.core.metrics import SpanFPreRecMetric
from fastNLP.core.trainer import Trainer
from fastNLP.io.config_io import ConfigLoader, ConfigSection
from fastNLP.models.sequence_labeling import AdvSeqLabel
from fastNLP.io.dataset_loader import ConllxDataLoader
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


def train(train_data_path, dev_data_path, checkpoint=None, save=None):
    # load config
    train_param = ConfigSection()
    model_param = ConfigSection()
    ConfigLoader().load_config(cfgfile, {"train": train_param, "model": model_param})
    print("config loaded")

    # Data Loader
    print("loading training set...")
    dataset = ConllxDataLoader().load(train_data_path, return_dataset=True)
    print("loading dev set...")
    dev_data = ConllxDataLoader().load(dev_data_path, return_dataset=True)
    print(dataset)
    print("================= dataset ready =====================")

    dataset.rename_field("tag", "truth")
    dev_data.rename_field("tag", "truth")

    vocab_proc = VocabIndexerProcessor("words", new_added_filed_name="word_seq")
    tag_proc = VocabIndexerProcessor("truth", is_input=True)
    seq_len_proc = SeqLenProcessor(field_name="word_seq", new_added_field_name="word_seq_origin_len", is_input=True)
    set_input_proc = SetInputProcessor("word_seq", "word_seq_origin_len")

    vocab_proc(dataset)
    tag_proc(dataset)
    seq_len_proc(dataset)

    # index dev set
    word_vocab, tag_vocab = vocab_proc.vocab, tag_proc.vocab
    dev_data.apply(lambda ins: [word_vocab.to_index(w) for w in ins["words"]], new_field_name="word_seq")
    dev_data.apply(lambda ins: [tag_vocab.to_index(w) for w in ins["truth"]], new_field_name="truth")
    dev_data.apply(lambda ins: len(ins["word_seq"]), new_field_name="word_seq_origin_len")

    # set input & target
    dataset.set_input("word_seq", "word_seq_origin_len", "truth")
    dev_data.set_input("word_seq", "word_seq_origin_len", "truth")
    dataset.set_target("truth", "word_seq_origin_len")
    dev_data.set_target("truth", "word_seq_origin_len")

    # dataset.set_is_target(tag_ids=True)
    model_param["vocab_size"] = vocab_proc.get_vocab_size()
    model_param["num_classes"] = tag_proc.get_vocab_size()
    print("vocab_size={}  num_classes={}".format(model_param["vocab_size"], model_param["num_classes"]))

    # define a model
    if checkpoint is None:
        # pre_trained = load_tencent_embed("/home/zyfeng/data/char_tencent_embedding.pkl", vocab_proc.vocab.word2idx)
        pre_trained = None
        model = AdvSeqLabel(model_param, id2words=None, emb=pre_trained)
        print(model)
    else:
        model = torch.load(checkpoint)

    # call trainer to train
    trainer = Trainer(dataset, model, loss=None, n_epochs=20, print_every=10, dev_data=dev_data,
                      metrics=SpanFPreRecMetric(tag_proc.vocab, pred="predict",
                                                target="truth",
                                                seq_lens="word_seq_origin_len"), metric_key="f", save_path=save,
                      use_tqdm=True)
    trainer.train(load_best_model=True)

    # save model & pipeline
    model_proc = ModelProcessor(model, seq_len_field_name="word_seq_origin_len")
    id2tag = Index2WordProcessor(tag_proc.vocab, "predict", "tag")

    pp = Pipeline([vocab_proc, seq_len_proc, set_input_proc, model_proc, id2tag])
    save_dict = {"pipeline": pp, "model": model, "tag_vocab": tag_proc.vocab}
    torch.save(save_dict, os.path.join(save, "model_pp.pkl"))
    print("pipeline saved")


def run_test(test_path):
    test_data = ConllxDataLoader().load(test_path, return_dataset=True)

    with open("model_pp_0117.pkl", "rb") as f:
        save_dict = torch.load(f)
    tag_vocab = save_dict["tag_vocab"]
    pipeline = save_dict["pipeline"]
    index_tag = IndexerProcessor(vocab=tag_vocab, field_name="tag", new_added_field_name="truth", is_input=False)
    pipeline.pipeline = [index_tag] + pipeline.pipeline

    pipeline(test_data)
    test_data.set_target("truth")
    prediction = test_data.field_arrays["predict"].content
    truth = test_data.field_arrays["truth"].content
    seq_len = test_data.field_arrays["word_seq_origin_len"].content

    # padding by hand
    max_length = max([len(seq) for seq in prediction])
    for idx in range(len(prediction)):
        prediction[idx] = list(prediction[idx]) + ([0] * (max_length - len(prediction[idx])))
        truth[idx] = list(truth[idx]) + ([0] * (max_length - len(truth[idx])))
    evaluator = SpanFPreRecMetric(tag_vocab=tag_vocab, pred="predict", target="truth",
                                  seq_lens="word_seq_origin_len")
    evaluator({"predict": torch.Tensor(prediction), "word_seq_origin_len": torch.Tensor(seq_len)},
              {"truth": torch.Tensor(truth)})
    test_result = evaluator.get_metric()
    f1 = round(test_result['f'] * 100, 2)
    pre = round(test_result['pre'] * 100, 2)
    rec = round(test_result['rec'] * 100, 2)

    return {"F1": f1, "precision": pre, "recall": rec}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="training conll file", default="/home/zyfeng/data/sample.conllx")
    parser.add_argument("--dev", type=str, help="dev conll file", default="/home/zyfeng/data/sample.conllx")
    parser.add_argument("--test", type=str, help="test conll file", default=None)
    parser.add_argument("--save", type=str, help="path to save", default=None)

    parser.add_argument("-c", "--restart", action="store_true", help="whether to continue training")
    parser.add_argument("-cp", "--checkpoint", type=str, help="checkpoint of the trained model")
    args = parser.parse_args()

    if args.test is not None:
        print(run_test(args.test))
    else:
        if args.restart is True:
            # 继续训练 python train_pos_tag.py -c -cp ./save/best_model.pkl
            if args.checkpoint is None:
                raise RuntimeError("Please provide the checkpoint. -cp ")
            train(args.train, args.dev, args.checkpoint, save=args.save)
        else:
            # 一次训练 python train_pos_tag.py
            train(args.train, args.dev, save=args.save)
