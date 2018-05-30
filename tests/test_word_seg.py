from action.tester import Tester
from action.trainer import Trainer
from loader.base_loader import BaseLoader
from model.word_seg_model import WordSegModel


def test_charlm():
    train_config = Trainer.TrainConfig(epochs=5, validate=False, save_when_better=False,
                                       log_per_step=10, log_validation=False, batch_size=254)
    trainer = Trainer(train_config)

    model = WordSegModel()

    train_data = BaseLoader("load_train", "./data_for_tests/cws_train").load_lines()

    trainer.train(model, train_data)

    trainer.save_model(model)

    test_config = Tester.TestConfig(save_output=False, validate_in_training=False,
                                    save_dev_input=False, save_loss=False, batch_size=254)
    tester = Tester(test_config)

    test_data = BaseLoader("load_test", "./data_for_tests/cws_test").load_lines()

    tester.test(model, test_data)


if __name__ == "__main__":
    test_charlm()
