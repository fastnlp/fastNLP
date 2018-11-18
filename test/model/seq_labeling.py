import os
import sys

sys.path.append("..")
import argparse
from fastNLP.io.config_loader import ConfigLoader, ConfigSection
from fastNLP.io.dataset_loader import BaseLoader
from fastNLP.io.model_saver import ModelSaver
from fastNLP.io.model_loader import ModelLoader
from fastNLP.core.tester import SeqLabelTester
from fastNLP.models.sequence_modeling import SeqLabeling
from fastNLP.core.predictor import SeqLabelInfer
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.dataset import SeqLabelDataSet, change_field_is_target
from fastNLP.core.metrics import SeqLabelEvaluator
from fastNLP.core.utils import save_pickle, load_pickle

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save", type=str, default="./seq_label/", help="path to save pickle files")
parser.add_argument("-t", "--train", type=str, default="../data_for_tests/people.txt",
                    help="path to the training data")
parser.add_argument("-c", "--config", type=str, default="../data_for_tests/config", help="path to the config file")
parser.add_argument("-m", "--model_name", type=str, default="seq_label_model.pkl", help="the name of the model")
parser.add_argument("-i", "--infer", type=str, default="../data_for_tests/people_infer.txt",
                    help="data used for inference")

args = parser.parse_args()
pickle_path = args.save
model_name = args.model_name
config_dir = args.config
data_path = args.train
data_infer_path = args.infer


def infer():
    # Load infer configuration, the same as test
    test_args = ConfigSection()
    ConfigLoader().load_config(config_dir, {"POS_infer": test_args})

    # fetch dictionary size and number of labels from pickle files
    word_vocab = load_pickle(pickle_path, "word2id.pkl")
    label_vocab = load_pickle(pickle_path, "label2id.pkl")
    test_args["vocab_size"] = len(word_vocab)
    test_args["num_classes"] = len(label_vocab)
    print("vocabularies loaded")

    # Define the same model
    model = SeqLabeling(test_args)
    print("model defined")

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, os.path.join(pickle_path, model_name))
    print("model loaded!")

    # Data Loader
    infer_data = SeqLabelDataSet(load_func=BaseLoader.load)
    infer_data.load(data_infer_path, vocabs={"word_vocab": word_vocab, "label_vocab": label_vocab}, infer=True)
    print("data set prepared")

    # Inference interface
    infer = SeqLabelInfer(pickle_path)
    results = infer.predict(model, infer_data)

    for res in results:
        print(res)
    print("Inference finished!")


def train_and_test():
    # Config Loader
    trainer_args = ConfigSection()
    model_args = ConfigSection()
    ConfigLoader().load_config(config_dir, {
        "test_seq_label_trainer": trainer_args, "test_seq_label_model": model_args})

    data_set = SeqLabelDataSet()
    data_set.load(data_path)
    train_set, dev_set = data_set.split(0.3, shuffle=True)
    model_args["vocab_size"] = len(data_set.word_vocab)
    model_args["num_classes"] = len(data_set.label_vocab)

    save_pickle(data_set.word_vocab, pickle_path, "word2id.pkl")
    save_pickle(data_set.label_vocab, pickle_path, "label2id.pkl")

    """
    trainer = SeqLabelTrainer(
        epochs=trainer_args["epochs"],
        batch_size=trainer_args["batch_size"],
        validate=False,
        use_cuda=trainer_args["use_cuda"],
        pickle_path=pickle_path,
        save_best_dev=trainer_args["save_best_dev"],
        model_name=model_name,
        optimizer=Optimizer("SGD", lr=0.01, momentum=0.9),
    )
    """

    # Model
    model = SeqLabeling(model_args)

    model.fit(train_set, dev_set,
              epochs=trainer_args["epochs"],
              batch_size=trainer_args["batch_size"],
              validate=False,
              use_cuda=trainer_args["use_cuda"],
              pickle_path=pickle_path,
              save_best_dev=trainer_args["save_best_dev"],
              model_name=model_name,
              optimizer=Optimizer("SGD", lr=0.01, momentum=0.9))

    # Start training
    # trainer.train(model, train_set, dev_set)
    print("Training finished!")

    # Saver
    saver = ModelSaver(os.path.join(pickle_path, model_name))
    saver.save_pytorch(model)
    print("Model saved!")

    del model

    change_field_is_target(dev_set, "truth", True)

    # Define the same model
    model = SeqLabeling(model_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, os.path.join(pickle_path, model_name))
    print("model loaded!")

    # Load test configuration
    tester_args = ConfigSection()
    ConfigLoader().load_config(config_dir, {"test_seq_label_tester": tester_args})

    # Tester
    tester = SeqLabelTester(batch_size=4,
                            use_cuda=False,
                            pickle_path=pickle_path,
                            model_name="seq_label_in_test.pkl",
                            evaluator=SeqLabelEvaluator()
                            )

    # Start testing with validation data
    tester.test(model, dev_set)
    print("model tested!")


if __name__ == "__main__":
    train_and_test()
    infer()
