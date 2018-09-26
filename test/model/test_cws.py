import os

from fastNLP.core.predictor import Predictor
from fastNLP.core.preprocess import Preprocessor, load_pickle
from fastNLP.core.tester import SeqLabelTester
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.loader.dataset_loader import TokenizeDatasetLoader, BaseLoader
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.models.sequence_modeling import SeqLabeling
from fastNLP.saver.model_saver import ModelSaver

data_name = "pku_training.utf8"
cws_data_path = "test/data_for_tests/cws_pku_utf_8"
pickle_path = "./save/"
data_infer_path = "test/data_for_tests/people_infer.txt"
config_path = "test/data_for_tests/config"

def infer():
    # Load infer configuration, the same as test
    test_args = ConfigSection()
    ConfigLoader("config.cfg").load_config(config_path, {"POS_infer": test_args})

    # fetch dictionary size and number of labels from pickle files
    word2index = load_pickle(pickle_path, "word2id.pkl")
    test_args["vocab_size"] = len(word2index)
    index2label = load_pickle(pickle_path, "class2id.pkl")
    test_args["num_classes"] = len(index2label)

    # Define the same model
    model = SeqLabeling(test_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, "./save/saved_model.pkl")
    print("model loaded!")

    # Data Loader
    raw_data_loader = BaseLoader(data_infer_path)
    infer_data = raw_data_loader.load_lines()

    # Inference interface
    infer = Predictor(pickle_path, "seq_label")
    results = infer.predict(model, infer_data)

    print(results)


def train_test():
    # Config Loader
    train_args = ConfigSection()
    ConfigLoader("config.cfg").load_config(config_path, {"POS_infer": train_args})

    # Data Loader
    loader = TokenizeDatasetLoader(cws_data_path)
    train_data = loader.load_pku()

    # Preprocessor
    p = Preprocessor(label_is_seq=True)
    data_train = p.run(train_data, pickle_path=pickle_path)
    train_args["vocab_size"] = p.vocab_size
    train_args["num_classes"] = p.num_classes

    # Trainer
    trainer = SeqLabelTrainer(**train_args.data)

    # Model
    model = SeqLabeling(train_args)

    # Start training
    trainer.train(model, data_train)

    # Saver
    saver = ModelSaver("./save/saved_model.pkl")
    saver.save_pytorch(model)

    del model, trainer, loader

    # Define the same model
    model = SeqLabeling(train_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, "./save/saved_model.pkl")

    # Load test configuration
    test_args = ConfigSection()
    ConfigLoader("config.cfg").load_config(config_path, {"POS_infer": test_args})

    # Tester
    tester = SeqLabelTester(**test_args.data)

    # Start testing
    tester.test(model, data_train)

    # print test results
    print(tester.show_metrics())


def test():
    os.makedirs("save", exist_ok=True)
    train_test()
    infer()
    os.system("rm -rf save")


if __name__ == "__main__":
    train_test()
    infer()
