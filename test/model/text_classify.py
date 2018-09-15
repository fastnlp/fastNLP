# Python: 3.5
# encoding: utf-8

import argparse
import os
import sys

sys.path.append("..")
from fastNLP.core.predictor import ClassificationInfer
from fastNLP.core.trainer import ClassificationTrainer
from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.loader.dataset_loader import ClassDatasetLoader
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.core.preprocess import ClassPreprocess
from fastNLP.models.cnn_text_classification import CNNText
from fastNLP.saver.model_saver import ModelSaver
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.loss import Loss

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save", type=str, default="./test_classification/", help="path to save pickle files")
parser.add_argument("-t", "--train", type=str, default="./data_for_tests/text_classify.txt",
                    help="path to the training data")
parser.add_argument("-c", "--config", type=str, default="./data_for_tests/config", help="path to the config file")
parser.add_argument("-m", "--model_name", type=str, default="classify_model.pkl", help="the name of the model")

args = parser.parse_args()
save_dir = args.save
train_data_dir = args.train
model_name = args.model_name
config_dir = args.config


def infer():
    # load dataset
    print("Loading data...")
    ds_loader = ClassDatasetLoader(train_data_dir)
    data = ds_loader.load()
    unlabeled_data = [x[0] for x in data]

    # pre-process data
    pre = ClassPreprocess()
    data = pre.run(data, pickle_path=save_dir)
    print("vocabulary size:", pre.vocab_size)
    print("number of classes:", pre.num_classes)

    model_args = ConfigSection()
    # TODO: load from config file
    model_args["vocab_size"] = pre.vocab_size
    model_args["num_classes"] = pre.num_classes
    # ConfigLoader.load_config(config_dir, {"text_class_model": model_args})

    # construct model
    print("Building model...")
    cnn = CNNText(model_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(cnn, os.path.join(save_dir, model_name))
    print("model loaded!")

    infer = ClassificationInfer(pickle_path=save_dir)
    results = infer.predict(cnn, unlabeled_data)
    print(results)


def train():
    train_args, model_args = ConfigSection(), ConfigSection()
    ConfigLoader.load_config(config_dir, {"text_class": train_args})

    # load dataset
    print("Loading data...")
    ds_loader = ClassDatasetLoader(train_data_dir)
    data = ds_loader.load()
    print(data[0])

    # pre-process data
    pre = ClassPreprocess()
    data_train = pre.run(data, pickle_path=save_dir)
    print("vocabulary size:", pre.vocab_size)
    print("number of classes:", pre.num_classes)

    model_args["num_classes"] = pre.num_classes
    model_args["vocab_size"] = pre.vocab_size

    # construct model
    print("Building model...")
    model = CNNText(model_args)

    # ConfigSaver().save_config(config_dir, {"text_class_model": model_args})

    # train
    print("Training...")

    # 1
    # trainer = ClassificationTrainer(train_args)

    # 2
    trainer = ClassificationTrainer(epochs=train_args["epochs"],
                                    batch_size=train_args["batch_size"],
                                    validate=train_args["validate"],
                                    use_cuda=train_args["use_cuda"],
                                    pickle_path=save_dir,
                                    save_best_dev=train_args["save_best_dev"],
                                    model_name=model_name,
                                    loss=Loss("cross_entropy"),
                                    optimizer=Optimizer("SGD", lr=0.001, momentum=0.9))
    trainer.train(model, data_train)

    print("Training finished!")

    saver = ModelSaver(os.path.join(save_dir, model_name))
    saver.save_pytorch(model)
    print("Model saved!")


if __name__ == "__main__":
    train()
    # infer()
