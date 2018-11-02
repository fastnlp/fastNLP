import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.loader.dataset_loader import PeopleDailyCorpusLoader, BaseLoader
from fastNLP.core.preprocess import load_pickle
from fastNLP.saver.model_saver import ModelSaver
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.core.tester import SeqLabelTester
from fastNLP.models.sequence_modeling import AdvSeqLabel
from fastNLP.core.predictor import SeqLabelInfer
from fastNLP.core.vocabulary import Vocabulary

# not in the file's dir
if len(os.path.dirname(__file__)) != 0:
    os.chdir(os.path.dirname(__file__))
datadir = "/home/zyfeng/data/"
cfgfile = './pos_tag.cfg'
data_name = "CWS_POS_TAG_NER_people_daily.txt"

pos_tag_data_path = os.path.join(datadir, data_name)
pickle_path = "save"
data_infer_path = os.path.join(datadir, "infer.utf8")


def infer():
    # Config Loader
    test_args = ConfigSection()
    ConfigLoader("config").load_config(cfgfile, {"POS_test": test_args})

    # fetch dictionary size and number of labels from pickle files
    word2index = load_pickle(pickle_path, "word2id.pkl")
    test_args["vocab_size"] = len(word2index)
    index2label = load_pickle(pickle_path, "class2id.pkl")
    test_args["num_classes"] = len(index2label)

    # Define the same model
    model = AdvSeqLabel(test_args)

    try:
        ModelLoader.load_pytorch(model, "./save/saved_model.pkl")
        print('model loaded!')
    except Exception as e:
        print('cannot load model!')
        raise

    # Data Loader
    raw_data_loader = BaseLoader(data_infer_path)
    infer_data = raw_data_loader.load_lines()
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
    ConfigLoader().load_config("./pos_tag.cfg", {"train": train_args, "test": test_args})

    # Data Loader
    loader = PeopleDailyCorpusLoader()
    data_set = loader.load(pos_tag_data_path)
    word_vocab = Vocabulary()
    label_vocab = Vocabulary()
    data_set.update_vocab(word_seq=word_vocab, label_seq=label_vocab)
    data_set.index_field("word_seq", word_vocab).index_field("label_seq", label_vocab)
    data_set.set_origin_len("word_seq")
    data_set.rename_field("label_seq", "truth").set_target(truth=False)
    data_train, data_dev = data_set.split(0.3, shuffle=True)
    train_args["vocab_size"] = len(word_vocab)
    train_args["num_classes"] = len(label_vocab)

    # Trainer
    trainer = SeqLabelTrainer(**train_args.data)

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
    saver = ModelSaver("./save/saved_model.pkl")
    saver.save_pytorch(model)
    print("Model saved!")


def test():
    # Config Loader
    test_args = ConfigSection()
    ConfigLoader("config").load_config(cfgfile, {"POS_test": test_args})

    # fetch dictionary size and number of labels from pickle files
    word2index = load_pickle(pickle_path, "word2id.pkl")
    test_args["vocab_size"] = len(word2index)
    index2label = load_pickle(pickle_path, "class2id.pkl")
    test_args["num_classes"] = len(index2label)

    # load dev data
    dev_data = load_pickle(pickle_path, "data_dev.pkl")

    # Define the same model
    model = AdvSeqLabel(test_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, "./save/saved_model.pkl")
    print("model loaded!")

    # Tester
    tester = SeqLabelTester(**test_args.data)

    # Start testing
    tester.test(model, dev_data)

    # print test results
    print(tester.show_metrics())
    print("model tested!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run a chinese word segmentation model')
    parser.add_argument('--mode', help='set the model\'s model', choices=['train', 'test', 'infer'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'infer':
        infer()
    else:
        print('no mode specified for model!')
        parser.print_help()
