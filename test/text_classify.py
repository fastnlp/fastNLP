# Python: 3.5
# encoding: utf-8

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

save_path = "./test_classification/"
data_dir = "./data_for_tests/"
train_file = 'text_classify.txt'
model_name = "model_class.pkl"


def infer():
    # load dataset
    print("Loading data...")
    ds_loader = ClassDatasetLoader("train", os.path.join(data_dir, train_file))
    data = ds_loader.load()
    unlabeled_data = [x[0] for x in data]

    # pre-process data
    pre = ClassPreprocess()
    vocab_size, n_classes = pre.run(data, pickle_path=save_path)
    print("vocabulary size:", vocab_size)
    print("number of classes:", n_classes)

    model_args = ConfigSection()
    ConfigLoader.load_config("data_for_tests/config", {"text_class_model": model_args})

    # construct model
    print("Building model...")
    cnn = CNNText(model_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(cnn, "./data_for_tests/saved_model.pkl")
    print("model loaded!")

    infer = ClassificationInfer(data_dir)
    results = infer.predict(cnn, unlabeled_data)
    print(results)


def train():
    train_args, model_args = ConfigSection(), ConfigSection()
    ConfigLoader.load_config("data_for_tests/config", {"text_class": train_args, "text_class_model": model_args})

    # load dataset
    print("Loading data...")
    ds_loader = ClassDatasetLoader("train", os.path.join(data_dir, train_file))
    data = ds_loader.load()
    print(data[0])

    # pre-process data
    pre = ClassPreprocess()
    data_train = pre.run(data, pickle_path=save_path)
    print("vocabulary size:", pre.vocab_size)
    print("number of classes:", pre.num_classes)

    # construct model
    print("Building model...")
    model = CNNText(model_args)

    # train
    print("Training...")

    trainer = ClassificationTrainer(train_args)
    trainer.train(model, data_train)

    print("Training finished!")

    saver = ModelSaver("./data_for_tests/saved_model.pkl")
    saver.save_pytorch(model)
    print("Model saved!")


if __name__ == "__main__":
    train()
    # infer()
