import sys

sys.path.append("..")

from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.loader.dataset_loader import TokenizeDatasetLoader, BaseLoader
from fastNLP.core.preprocess import SeqLabelPreprocess, load_pickle
from fastNLP.saver.model_saver import ModelSaver
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.core.tester import SeqLabelTester
from fastNLP.models.sequence_modeling import SeqLabeling
from fastNLP.core.predictor import Predictor

data_name = "pku_training.utf8"
# cws_data_path = "/home/zyfeng/Desktop/data/pku_training.utf8"
cws_data_path = "data_for_tests/cws_pku_utf_8"
pickle_path = "data_for_tests"
data_infer_path = "data_for_tests/people_infer.txt"


def infer():
    # Load infer configuration, the same as test
    test_args = ConfigSection()
    ConfigLoader("config.cfg").load_config("./data_for_tests/config", {"POS_test": test_args})

    # fetch dictionary size and number of labels from pickle files
    word2index = load_pickle(pickle_path, "word2id.pkl")
    test_args["vocab_size"] = len(word2index)
    index2label = load_pickle(pickle_path, "class2id.pkl")
    test_args["num_classes"] = len(index2label)

    # Define the same model
    model = SeqLabeling(test_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, "./data_for_tests/saved_model.pkl")
    print("model loaded!")

    # Data Loader
    raw_data_loader = BaseLoader(data_infer_path)
    infer_data = raw_data_loader.load_lines()
    """
        Transform strings into list of list of strings. 
        [
            [word_11, word_12, ...],
            [word_21, word_22, ...],
            ...
        ]
        In this case, each line in "people_infer.txt" is already a sentence. So load_lines() just splits them.
    """

    # Inference interface
    infer = Predictor(pickle_path)
    results = infer.predict(model, infer_data)

    print(results)
    print("Inference finished!")


def train_test():
    # Config Loader
    train_args = ConfigSection()
    ConfigLoader("config.cfg").load_config("./data_for_tests/config", {"POS": train_args})

    # Data Loader
    loader = TokenizeDatasetLoader(cws_data_path)
    train_data = loader.load_pku()

    # Preprocessor
    p = SeqLabelPreprocess()
    data_train = p.run(train_data, pickle_path=pickle_path)
    train_args["vocab_size"] = p.vocab_size
    train_args["num_classes"] = p.num_classes

    # Trainer
    trainer = SeqLabelTrainer(**train_args.data)

    # Model
    model = SeqLabeling(train_args)

    # Start training
    trainer.train(model, data_train)
    print("Training finished!")

    # Saver
    saver = ModelSaver("./data_for_tests/saved_model.pkl")
    saver.save_pytorch(model)
    print("Model saved!")

    del model, trainer, loader

    # Define the same model
    model = SeqLabeling(train_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, "./data_for_tests/saved_model.pkl")
    print("model loaded!")

    # Load test configuration
    test_args = ConfigSection()
    ConfigLoader("config.cfg").load_config("./data_for_tests/config", {"POS_test": test_args})

    # Tester
    tester = SeqLabelTester(**test_args.data)

    # Start testing
    tester.test(model, data_train)

    # print test results
    print(tester.show_metrics())
    print("model tested!")


if __name__ == "__main__":
    train_test()
    infer()
