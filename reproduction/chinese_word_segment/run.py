import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.loader.dataset_loader import BaseLoader, TokenizeDataSetLoader
from fastNLP.core.preprocess import load_pickle
from fastNLP.saver.model_saver import ModelSaver
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.core.tester import SeqLabelTester
from fastNLP.models.sequence_modeling import AdvSeqLabel
from fastNLP.core.predictor import SeqLabelInfer
from fastNLP.core.dataset import DataSet
from fastNLP.core.preprocess import save_pickle
from fastNLP.core.metrics import SeqLabelEvaluator

# not in the file's dir
if len(os.path.dirname(__file__)) != 0:
    os.chdir(os.path.dirname(__file__))
datadir = "/home/zyfeng/data/"
cfgfile = './cws.cfg'

cws_data_path = os.path.join(datadir, "pku_training.utf8")
pickle_path = "save"
data_infer_path = os.path.join(datadir, "infer.utf8")


def infer():
    # Config Loader
    test_args = ConfigSection()
    ConfigLoader().load_config(cfgfile, {"POS_test": test_args})

    # fetch dictionary size and number of labels from pickle files
    word2index = load_pickle(pickle_path, "word2id.pkl")
    test_args["vocab_size"] = len(word2index)
    index2label = load_pickle(pickle_path, "label2id.pkl")
    test_args["num_classes"] = len(index2label)

    # Define the same model
    model = AdvSeqLabel(test_args)

    try:
        ModelLoader.load_pytorch(model, "./save/trained_model.pkl")
        print('model loaded!')
    except Exception as e:
        print('cannot load model!')
        raise

    # Data Loader
    infer_data = SeqLabelDataSet(load_func=BaseLoader.load_lines)
    infer_data.load(data_infer_path, vocabs={"word_vocab": word2index}, infer=True)
    print('data loaded')

    # Inference interface
    infer = SeqLabelInfer(pickle_path)
    results = infer.predict(model, infer_data)

    print(results)
    print("Inference finished!")


def train():
    # Config Loader
    train_args = ConfigSection()
    test_args = ConfigSection()
    ConfigLoader().load_config(cfgfile, {"train": train_args, "test": test_args})

    print("loading data set...")
    data = SeqLabelDataSet(load_func=TokenizeDataSetLoader.load)
    data.load(cws_data_path)
    data_train, data_dev = data.split(ratio=0.3)
    train_args["vocab_size"] = len(data.word_vocab)
    train_args["num_classes"] = len(data.label_vocab)
    print("vocab size={}, num_classes={}".format(len(data.word_vocab), len(data.label_vocab)))

    change_field_is_target(data_dev, "truth", True)
    save_pickle(data_dev, "./save/", "data_dev.pkl")
    save_pickle(data.word_vocab, "./save/", "word2id.pkl")
    save_pickle(data.label_vocab, "./save/", "label2id.pkl")

    # Trainer
    trainer = SeqLabelTrainer(epochs=train_args["epochs"], batch_size=train_args["batch_size"],
                              validate=train_args["validate"],
                              use_cuda=train_args["use_cuda"], pickle_path=train_args["pickle_path"],
                              save_best_dev=True, print_every_step=10, model_name="trained_model.pkl",
                              evaluator=SeqLabelEvaluator())

    # Model
    model = AdvSeqLabel(train_args)
    try:
        ModelLoader.load_pytorch(model, "./save/saved_model.pkl")
        print('model parameter loaded!')
    except Exception as e:
        print("No saved model. Continue.")
        pass

    # Start training
    trainer.train(model, data_train, data_dev)
    print("Training finished!")

    # Saver
    saver = ModelSaver("./save/trained_model.pkl")
    saver.save_pytorch(model)
    print("Model saved!")


def predict():
    # Config Loader
    test_args = ConfigSection()
    ConfigLoader().load_config(cfgfile, {"POS_test": test_args})

    # fetch dictionary size and number of labels from pickle files
    word2index = load_pickle(pickle_path, "word2id.pkl")
    test_args["vocab_size"] = len(word2index)
    index2label = load_pickle(pickle_path, "label2id.pkl")
    test_args["num_classes"] = len(index2label)

    # load dev data
    dev_data = load_pickle(pickle_path, "data_dev.pkl")

    # Define the same model
    model = AdvSeqLabel(test_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, "./save/trained_model.pkl")
    print("model loaded!")

    # Tester
    test_args["evaluator"] = SeqLabelEvaluator()
    tester = SeqLabelTester(**test_args.data)

    # Start testing
    tester.test(model, dev_data)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Run a chinese word segmentation model')
    parser.add_argument('--mode', help='set the model\'s model', choices=['train', 'test', 'infer'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        predict()
    elif args.mode == 'infer':
        infer()
    else:
        print('no mode specified for model!')
        parser.print_help()
