from fastNLP.core.preprocess import SeqLabelPreprocess
from fastNLP.core.tester import SeqLabelTester
from fastNLP.loader.config_loader import ConfigSection, ConfigLoader
from fastNLP.loader.dataset_loader import TokenizeDatasetLoader
from fastNLP.models.sequence_modeling import SeqLabeling

data_name = "pku_training.utf8"
cws_data_path = "/home/zyfeng/Desktop/data/pku_training.utf8"
pickle_path = "data_for_tests"


def foo():
    loader = TokenizeDatasetLoader(data_name, "./data_for_tests/cws_pku_utf_8")
    train_data = loader.load_pku()

    train_args = ConfigSection()
    ConfigLoader("config.cfg", "").load_config("./data_for_tests/config", {"POS": train_args})

    # Preprocessor
    p = SeqLabelPreprocess(train_data, pickle_path)
    train_args["vocab_size"] = p.vocab_size
    train_args["num_classes"] = p.num_classes

    model = SeqLabeling(train_args)

    valid_args = {"save_output": True, "validate_in_training": True, "save_dev_input": True,
                  "save_loss": True, "batch_size": 8, "pickle_path": "./data_for_tests/",
                  "use_cuda": True}
    validator = SeqLabelTester(valid_args)

    print("start validation.")
    validator.test(model)
    print(validator.show_matrices())


if __name__ == "__main__":
    foo()
