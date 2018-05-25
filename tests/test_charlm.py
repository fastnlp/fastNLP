from action.tester import Tester
from action.trainer import Trainer
from loader.base_loader import ToyLoader0
from model.char_language_model import CharLM


def test_charlm():
    train_config = Trainer.TrainConfig(epochs=1, validate=True, save_when_better=True,
                                       log_per_step=10, log_validation=True, batch_size=160)
    trainer = Trainer(train_config)

    model = CharLM(lstm_batch_size=16, lstm_seq_len=10)

    train_data = ToyLoader0("load_train", "./data_for_tests/charlm.txt").load()
    valid_data = ToyLoader0("load_valid", "./data_for_tests/charlm.txt").load()

    trainer.train(model, train_data, valid_data)

    trainer.save_model(model)

    test_config = Tester.TestConfig(save_output=True, validate_in_training=True,
                                    save_dev_input=True, save_loss=True, batch_size=160)
    tester = Tester(test_config)

    test_data = ToyLoader0("load_test", "./data_for_tests/charlm.txt").load()

    tester.test(model, test_data)


if __name__ == "__main__":
    test_charlm()
