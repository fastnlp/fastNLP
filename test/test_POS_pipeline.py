import sys

sys.path.append("..")

from fastNLP.action.trainer import POSTrainer
from fastNLP.loader.dataset_loader import POSDatasetLoader
from fastNLP.loader.preprocess import POSPreprocess
from fastNLP.models.sequence_modeling import SeqLabeling

data_name = "people.txt"
data_path = "data_for_tests/people.txt"
pickle_path = "data_for_tests"

if __name__ == "__main__":
    # Data Loader
    pos = POSDatasetLoader(data_name, data_path)
    train_data = pos.load_lines()

    # Preprocessor
    p = POSPreprocess(train_data, pickle_path)
    vocab_size = p.vocab_size
    num_classes = p.num_classes

    # Trainer
    train_args = {"epochs": 20, "batch_size": 1, "num_classes": num_classes,
                  "vocab_size": vocab_size, "pickle_path": pickle_path, "validate": True}
    trainer = POSTrainer(train_args)

    # Model
    model = SeqLabeling(100, 1, num_classes, vocab_size, bi_direction=True)

    # Start training
    trainer.train(model)

    print("Training finished!")
