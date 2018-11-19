import os

from fastNLP.core.metrics import SeqLabelEvaluator
from fastNLP.core.predictor import Predictor
from fastNLP.core.tester import Tester
from fastNLP.core.trainer import Trainer
from fastNLP.core.utils import save_pickle, load_pickle
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.io.config_loader import ConfigLoader, ConfigSection
from fastNLP.io.dataset_loader import TokenizeDataSetLoader, RawDataSetLoader
from fastNLP.io.model_loader import ModelLoader
from fastNLP.io.model_saver import ModelSaver
from fastNLP.models.sequence_modeling import SeqLabeling

data_name = "pku_training.utf8"
cws_data_path = "./test/data_for_tests/cws_pku_utf_8"
pickle_path = "./save/"
data_infer_path = "./test/data_for_tests/people_infer.txt"
config_path = "./test/data_for_tests/config"

def infer():
    # Load infer configuration, the same as test
    test_args = ConfigSection()
    ConfigLoader().load_config(config_path, {"POS_infer": test_args})

    # fetch dictionary size and number of labels from pickle files
    word2index = load_pickle(pickle_path, "word2id.pkl")
    test_args["vocab_size"] = len(word2index)
    index2label = load_pickle(pickle_path, "label2id.pkl")
    test_args["num_classes"] = len(index2label)

    # Define the same model
    model = SeqLabeling(test_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, "./save/saved_model.pkl")
    print("model loaded!")

    # Load infer data
    infer_data = RawDataSetLoader().load(data_infer_path)
    infer_data.index_field("word_seq", word2index)
    infer_data.set_origin_len("word_seq")
    # inference
    infer = Predictor(pickle_path)
    results = infer.predict(model, infer_data)
    print(results)


def train_test():
    # Config Loader
    train_args = ConfigSection()
    ConfigLoader().load_config(config_path, {"POS_infer": train_args})

    # define dataset
    data_train = TokenizeDataSetLoader().load(cws_data_path)
    word_vocab = Vocabulary()
    label_vocab = Vocabulary()
    data_train.update_vocab(word_seq=word_vocab, label_seq=label_vocab)
    data_train.index_field("word_seq", word_vocab).index_field("label_seq", label_vocab)
    data_train.set_origin_len("word_seq")
    data_train.rename_field("label_seq", "truth").set_target(truth=False)
    train_args["vocab_size"] = len(word_vocab)
    train_args["num_classes"] = len(label_vocab)

    save_pickle(word_vocab, pickle_path, "word2id.pkl")
    save_pickle(label_vocab, pickle_path, "label2id.pkl")

    # Trainer
    trainer = Trainer(**train_args.data)

    # Model
    model = SeqLabeling(train_args)

    # Start training
    trainer.train(model, data_train)

    # Saver
    saver = ModelSaver("./save/saved_model.pkl")
    saver.save_pytorch(model)

    del model, trainer

    # Define the same model
    model = SeqLabeling(train_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, "./save/saved_model.pkl")

    # Load test configuration
    test_args = ConfigSection()
    ConfigLoader().load_config(config_path, {"POS_infer": test_args})
    test_args["evaluator"] = SeqLabelEvaluator()

    # Tester
    tester = Tester(**test_args.data)

    # Start testing
    data_train.set_target(truth=True)
    tester.test(model, data_train)


def test():
    os.makedirs("save", exist_ok=True)
    train_test()
    infer()
    os.system("rm -rf save")


if __name__ == "__main__":
    train_test()
    infer()
