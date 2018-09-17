import os
import sys
sys.path.append("..")
import argparse
from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.loader.dataset_loader import POSDatasetLoader, BaseLoader
from fastNLP.core.preprocess import SeqLabelPreprocess, load_pickle
from fastNLP.saver.model_saver import ModelSaver
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.core.tester import SeqLabelTester
from fastNLP.models.sequence_modeling import SeqLabeling
from fastNLP.core.predictor import SeqLabelInfer
from fastNLP.core.optimizer import Optimizer

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
    ConfigLoader("config.cfg").load_config(config_dir, {"POS_infer": test_args})

    # fetch dictionary size and number of labels from pickle files
    word2index = load_pickle(pickle_path, "word2id.pkl")
    test_args["vocab_size"] = len(word2index)
    index2label = load_pickle(pickle_path, "id2class.pkl")
    test_args["num_classes"] = len(index2label)

    # Define the same model
    model = SeqLabeling(test_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, os.path.join(pickle_path, model_name))
    print("model loaded!")

    # Data Loader
    raw_data_loader = BaseLoader(data_infer_path)
    infer_data = raw_data_loader.load_lines()

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
    ConfigLoader("config.cfg").load_config(config_dir, {
        "test_seq_label_trainer": trainer_args, "test_seq_label_model": model_args})

    # Data Loader
    pos_loader = POSDatasetLoader(data_path)
    train_data = pos_loader.load_lines()

    # Preprocessor
    p = SeqLabelPreprocess()
    data_train, data_dev = p.run(train_data, pickle_path=pickle_path, train_dev_split=0.5)
    model_args["vocab_size"] = p.vocab_size
    model_args["num_classes"] = p.num_classes

    # Trainer: two definition styles
    # 1
    # trainer = SeqLabelTrainer(trainer_args.data)

    # 2
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

    # Model
    model = SeqLabeling(model_args)

    # Start training
    trainer.train(model, data_train, data_dev)
    print("Training finished!")

    # Saver
    saver = ModelSaver(os.path.join(pickle_path, model_name))
    saver.save_pytorch(model)
    print("Model saved!")

    del model, trainer, pos_loader

    # Define the same model
    model = SeqLabeling(model_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, os.path.join(pickle_path, model_name))
    print("model loaded!")

    # Load test configuration
    tester_args = ConfigSection()
    ConfigLoader("config.cfg").load_config(config_dir, {"test_seq_label_tester": tester_args})

    # Tester
    tester = SeqLabelTester(save_output=False,
                            save_loss=True,
                            save_best_dev=False,
                            batch_size=4,
                            use_cuda=False,
                            pickle_path=pickle_path,
                            model_name="seq_label_in_test.pkl",
                            print_every_step=1
                            )

    # Start testing with validation data
    tester.test(model, data_dev)

    # print test results
    print(tester.show_metrics())
    print("model tested!")


if __name__ == "__main__":
    train_and_test()
    # infer()
