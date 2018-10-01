import os

from fastNLP.core.dataset import SeqLabelDataSet, change_field_is_target
from fastNLP.core.metrics import SeqLabelEvaluator
from fastNLP.core.predictor import SeqLabelInfer
from fastNLP.core.preprocess import save_pickle, load_pickle
from fastNLP.core.tester import SeqLabelTester
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.loader.dataset_loader import TokenizeDataSetLoader, BaseLoader
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.models.sequence_modeling import SeqLabeling
from fastNLP.saver.model_saver import ModelSaver

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
    infer_data = SeqLabelDataSet(load_func=BaseLoader.load)
    infer_data.load(data_infer_path, vocabs={"word_vocab": word2index}, infer=True)

    # inference
    infer = SeqLabelInfer(pickle_path)
    results = infer.predict(model, infer_data)
    print(results)


def train_test():
    # Config Loader
    train_args = ConfigSection()
    ConfigLoader().load_config(config_path, {"POS_infer": train_args})

    # define dataset
    data_train = SeqLabelDataSet(load_func=TokenizeDataSetLoader.load)
    data_train.load(cws_data_path)
    train_args["vocab_size"] = len(data_train.word_vocab)
    train_args["num_classes"] = len(data_train.label_vocab)

    save_pickle(data_train.word_vocab, pickle_path, "word2id.pkl")
    save_pickle(data_train.label_vocab, pickle_path, "label2id.pkl")

    # Trainer
    trainer = SeqLabelTrainer(**train_args.data)

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
    tester = SeqLabelTester(**test_args.data)

    # Start testing
    change_field_is_target(data_train, "truth", True)
    tester.test(model, data_train)


def test():
    os.makedirs("save", exist_ok=True)
    train_test()
    infer()
    os.system("rm -rf save")


if __name__ == "__main__":
    train_test()
    infer()
