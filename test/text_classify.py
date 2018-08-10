# Python: 3.5
# encoding: utf-8

import os

from fastNLP.core.inference import ClassificationInfer
from fastNLP.core.trainer import ClassificationTrainer
from fastNLP.loader.dataset_loader import ClassDatasetLoader
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.loader.preprocess import ClassPreprocess
from fastNLP.models.cnn_text_classification import CNNText
from fastNLP.saver.model_saver import ModelSaver

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
    pre = ClassPreprocess(data_dir)
    vocab_size, n_classes = pre.process(data, "data_train.pkl")
    print("vocabulary size:", vocab_size)
    print("number of classes:", n_classes)

    # construct model
    print("Building model...")
    cnn = CNNText(class_num=n_classes, embed_num=vocab_size)
    # Dump trained parameters into the model
    ModelLoader.load_pytorch(cnn, "./data_for_tests/saved_model.pkl")
    print("model loaded!")

    infer = ClassificationInfer(data_dir)
    results = infer.predict(cnn, unlabeled_data)
    print(results)


def train():
    # load dataset
    print("Loading data...")
    ds_loader = ClassDatasetLoader("train", os.path.join(data_dir, train_file))
    data = ds_loader.load()
    print(data[0])

    # pre-process data
    pre = ClassPreprocess(data_dir)
    vocab_size, n_classes = pre.process(data, "data_train.pkl")
    print("vocabulary size:", vocab_size)
    print("number of classes:", n_classes)

    # construct model
    print("Building model...")
    cnn = CNNText(class_num=n_classes, embed_num=vocab_size)

    # train
    print("Training...")
    train_args = {
        "epochs": 1,
        "batch_size": 10,
        "pickle_path": data_dir,
        "validate": False,
        "save_best_dev": False,
        "model_saved_path": "./data_for_tests/",
        "use_cuda": True
    }
    trainer = ClassificationTrainer(train_args)
    trainer.train(cnn)

    print("Training finished!")

    saver = ModelSaver("./data_for_tests/saved_model.pkl")
    saver.save_pytorch(cnn)
    print("Model saved!")


if __name__ == "__main__":
    # train()
    infer()
