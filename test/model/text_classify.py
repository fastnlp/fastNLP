# Python: 3.5
# encoding: utf-8

import argparse
import os
import sys

sys.path.append("..")
from fastNLP.core.predictor import ClassificationInfer
from fastNLP.core.trainer import ClassificationTrainer
from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.loader.dataset_loader import ClassDataSetLoader
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.models.cnn_text_classification import CNNText
from fastNLP.saver.model_saver import ModelSaver
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.loss import Loss
from fastNLP.core.dataset import TextClassifyDataSet
from fastNLP.core.preprocess import save_pickle, load_pickle

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save", type=str, default="./test_classification/", help="path to save pickle files")
parser.add_argument("-t", "--train", type=str, default="../data_for_tests/text_classify.txt",
                    help="path to the training data")
parser.add_argument("-c", "--config", type=str, default="../data_for_tests/config", help="path to the config file")
parser.add_argument("-m", "--model_name", type=str, default="classify_model.pkl", help="the name of the model")

args = parser.parse_args()
save_dir = args.save
train_data_dir = args.train
model_name = args.model_name
config_dir = args.config


def infer():
    # load dataset
    print("Loading data...")
    word_vocab = load_pickle(save_dir, "word2id.pkl")
    label_vocab = load_pickle(save_dir, "label2id.pkl")
    print("vocabulary size:", len(word_vocab))
    print("number of classes:", len(label_vocab))

    infer_data = TextClassifyDataSet(loader=ClassDataSetLoader())
    infer_data.load(train_data_dir, vocabs={"word_vocab": word_vocab, "label_vocab": label_vocab})

    model_args = ConfigSection()
    model_args["vocab_size"] = len(word_vocab)
    model_args["num_classes"] = len(label_vocab)
    ConfigLoader.load_config(config_dir, {"text_class_model": model_args})

    # construct model
    print("Building model...")
    cnn = CNNText(model_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(cnn, os.path.join(save_dir, model_name))
    print("model loaded!")

    infer = ClassificationInfer(pickle_path=save_dir)
    results = infer.predict(cnn, infer_data)
    print(results)


def train():
    train_args, model_args = ConfigSection(), ConfigSection()
    ConfigLoader.load_config(config_dir, {"text_class": train_args})

    # load dataset
    print("Loading data...")
    data = TextClassifyDataSet(loader=ClassDataSetLoader())
    data.load(train_data_dir)

    print("vocabulary size:", len(data.word_vocab))
    print("number of classes:", len(data.label_vocab))
    save_pickle(data.word_vocab, save_dir, "word2id.pkl")
    save_pickle(data.label_vocab, save_dir, "label2id.pkl")

    model_args["num_classes"] = len(data.label_vocab)
    model_args["vocab_size"] = len(data.word_vocab)

    # construct model
    print("Building model...")
    model = CNNText(model_args)

    # train
    print("Training...")
    trainer = ClassificationTrainer(epochs=train_args["epochs"],
                                    batch_size=train_args["batch_size"],
                                    validate=train_args["validate"],
                                    use_cuda=train_args["use_cuda"],
                                    pickle_path=save_dir,
                                    save_best_dev=train_args["save_best_dev"],
                                    model_name=model_name,
                                    loss=Loss("cross_entropy"),
                                    optimizer=Optimizer("SGD", lr=0.001, momentum=0.9))
    trainer.train(model, data)

    print("Training finished!")

    saver = ModelSaver(os.path.join(save_dir, model_name))
    saver.save_pytorch(model)
    print("Model saved!")


if __name__ == "__main__":
    train()
    infer()
