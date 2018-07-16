import sys

sys.path.append("..")

from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.action.trainer import POSTrainer
from fastNLP.loader.dataset_loader import POSDatasetLoader
from fastNLP.loader.preprocess import POSPreprocess
from fastNLP.models.sequence_modeling import SeqLabeling

data_name = "people.txt"
data_path = "data_for_tests/people.txt"
pickle_path = "data_for_tests"

if __name__ == "__main__":
    train_args = ConfigSection()
    ConfigLoader("config.cfg", "").load_config("./data_for_tests/config", {"POS": train_args})

    # Data Loader
    pos = POSDatasetLoader(data_name, data_path)
    train_data = pos.load_lines()

    # Preprocessor
    p = POSPreprocess(train_data, pickle_path)
    vocab_size = p.vocab_size
    num_classes = p.num_classes

    train_args["vocab_size"] = vocab_size
    train_args["num_classes"] = num_classes

    trainer = POSTrainer(train_args)

    # Model
    model = SeqLabeling(100, 1, num_classes, vocab_size, bi_direction=True)

    # Start training
    trainer.train(model)

    print("Training finished!")
