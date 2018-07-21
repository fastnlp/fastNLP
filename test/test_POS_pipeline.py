import sys

sys.path.append("..")

from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.action.trainer import POSTrainer
from fastNLP.loader.dataset_loader import POSDatasetLoader
from fastNLP.loader.preprocess import POSPreprocess
from fastNLP.saver.model_saver import ModelSaver
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.action.tester import POSTester
from fastNLP.models.sequence_modeling import SeqLabeling
from fastNLP.action.inference import Inference

data_name = "people.txt"
data_path = "data_for_tests/people.txt"
pickle_path = "data_for_tests"


def test_infer():
    # Define the same model
    model = SeqLabeling(hidden_dim=train_args["rnn_hidden_units"], rnn_num_layer=train_args["rnn_layers"],
                        num_classes=train_args["num_classes"], vocab_size=train_args["vocab_size"],
                        word_emb_dim=train_args["word_emb_dim"], bi_direction=train_args["rnn_bi_direction"],
                        rnn_mode="gru", dropout=train_args["dropout"], use_crf=train_args["use_crf"])

    # Dump trained parameters into the model
    ModelLoader("arbitrary_name", "./saved_model.pkl").load_pytorch(model)
    print("model loaded!")

    # Data Loader
    pos_loader = POSDatasetLoader(data_name, data_path)
    infer_data = pos_loader.load_lines()

    # Preprocessor
    POSPreprocess(infer_data, pickle_path)

    # Inference interface
    infer = Inference()
    results = infer.predict(model, infer_data)


if __name__ == "__main__":
    # Config Loader
    train_args = ConfigSection()
    ConfigLoader("config.cfg", "").load_config("./data_for_tests/config", {"POS": train_args})

    # Data Loader
    pos_loader = POSDatasetLoader(data_name, data_path)
    train_data = pos_loader.load_lines()

    # Preprocessor
    p = POSPreprocess(train_data, pickle_path)
    train_args["vocab_size"] = p.vocab_size
    train_args["num_classes"] = p.num_classes

    # Trainer
    trainer = POSTrainer(train_args)

    # Model
    model = SeqLabeling(hidden_dim=train_args["rnn_hidden_units"], rnn_num_layer=train_args["rnn_layers"],
                        num_classes=train_args["num_classes"], vocab_size=train_args["vocab_size"],
                        word_emb_dim=train_args["word_emb_dim"], bi_direction=train_args["rnn_bi_direction"],
                        rnn_mode="gru", dropout=train_args["dropout"], use_crf=train_args["use_crf"])

    # Start training
    trainer.train(model)
    print("Training finished!")

    # Saver
    saver = ModelSaver("./saved_model.pkl")
    saver.save_pytorch(model)
    print("Model saved!")

    del model, trainer, pos_loader

    # Define the same model
    model = SeqLabeling(hidden_dim=train_args["rnn_hidden_units"], rnn_num_layer=train_args["rnn_layers"],
                        num_classes=train_args["num_classes"], vocab_size=train_args["vocab_size"],
                        word_emb_dim=train_args["word_emb_dim"], bi_direction=train_args["rnn_bi_direction"],
                        rnn_mode="gru", dropout=train_args["dropout"], use_crf=train_args["use_crf"])

    # Dump trained parameters into the model
    ModelLoader("arbitrary_name", "./saved_model.pkl").load_pytorch(model)
    print("model loaded!")

    # Load test configuration
    test_args = ConfigSection()
    ConfigLoader("config.cfg", "").load_config("./data_for_tests/config", {"POS_test": test_args})

    # Tester
    tester = POSTester(test_args)

    # Start testing
    tester.test(model)

    # print test results
    print(tester.show_matrices())
    print("model tested!")
