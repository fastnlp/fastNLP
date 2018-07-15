import sys
sys.path.append("..")
from fastNLP.action.trainer import POSTrainer
from fastNLP.loader.dataset_loader import POSDatasetLoader
from fastNLP.loader.preprocess import POSPreprocess
from fastNLP.saver.model_saver import ModelSaver
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.action.tester import POSTester
from fastNLP.models.sequence_modeling import SeqLabeling

data_name = "people.txt"
data_path = "data_for_tests/people.txt"
pickle_path = "data_for_tests"

if __name__ == "__main__":
    # Data Loader
    pos_loader = POSDatasetLoader(data_name, data_path)
    train_data = pos_loader.load_lines()

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

    saver = ModelSaver("./saved_model.pkl")
    saver.save_pytorch(model)
    print("Model saved!")

    del model, trainer, pos_loader

    model = SeqLabeling(100, 1, num_classes, vocab_size, bi_direction=True)
    ModelLoader("xxx", "./saved_model.pkl").load_pytorch(model)
    print("model loaded!")

    test_args = {"save_output": True, "validate_in_training": False, "save_dev_input": False,
                 "save_loss": True, "batch_size": 1, "pickle_path": pickle_path}
    tester = POSTester(test_args)
    tester.test(model)
    print("model tested!")
