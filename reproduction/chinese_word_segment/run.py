import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.loader.dataset_loader import TokenizeDatasetLoader, BaseLoader
from fastNLP.loader.preprocess import POSPreprocess, load_pickle
from fastNLP.saver.model_saver import ModelSaver
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.core.tester import SeqLabelTester
from fastNLP.models.sequence_modeling import AdvSeqLabel
from fastNLP.core.inference import SeqLabelInfer

# not in the file's dir
if len(os.path.dirname(__file__)) != 0:
    os.chdir(os.path.dirname(__file__))
datadir = 'icwb2-data'
cfgfile = 'cws.cfg'
data_name = "pku_training.utf8"

cws_data_path = os.path.join(datadir, "training/pku_training.utf8")
pickle_path = "save"
data_infer_path = os.path.join(datadir, "infer.utf8")

def infer():
    # Config Loader
    test_args = ConfigSection()
    ConfigLoader("config", "").load_config(cfgfile, {"POS_test": test_args})

    # fetch dictionary size and number of labels from pickle files
    word2index = load_pickle(pickle_path, "word2id.pkl")
    test_args["vocab_size"] = len(word2index)
    index2label = load_pickle(pickle_path, "id2class.pkl")
    test_args["num_classes"] = len(index2label)


    # Define the same model
    model = AdvSeqLabel(test_args)

    try:
        ModelLoader.load_pytorch(model, "./save/saved_model.pkl")
        print('model loaded!')
    except Exception as e:
        print('cannot load model!')
        raise

    # Data Loader
    raw_data_loader = BaseLoader(data_name, data_infer_path)
    infer_data = raw_data_loader.load_lines()
    print('data loaded')

    # Inference interface
    infer = SeqLabelInfer(pickle_path)
    results = infer.predict(model, infer_data)

    print(results)
    print("Inference finished!")


def train():
    # Config Loader
    train_args = ConfigSection()
    test_args = ConfigSection()
    ConfigLoader("good_name", "good_path").load_config(cfgfile, {"train": train_args, "test": test_args})

    # Data Loader
    loader = TokenizeDatasetLoader(data_name, cws_data_path)
    train_data = loader.load_pku()

    # Preprocessor
    p = POSPreprocess(train_data, pickle_path, train_dev_split=0.3)
    train_args["vocab_size"] = p.vocab_size
    train_args["num_classes"] = p.num_classes

    # Trainer
    trainer = SeqLabelTrainer(**train_args.data)

    # Model
    model = AdvSeqLabel(train_args)
    try:
        ModelLoader.load_pytorch(model, "./save/saved_model.pkl")
        print('model parameter loaded!')
    except Exception as e:
        pass
        
    # Start training
    trainer.train(model)
    print("Training finished!")

    # Saver
    saver = ModelSaver("./save/saved_model.pkl")
    saver.save_pytorch(model)
    print("Model saved!")


def test():
    # Config Loader
    test_args = ConfigSection()
    ConfigLoader("config", "").load_config(cfgfile, {"POS_test": test_args})

    # fetch dictionary size and number of labels from pickle files
    word2index = load_pickle(pickle_path, "word2id.pkl")
    test_args["vocab_size"] = len(word2index)
    index2label = load_pickle(pickle_path, "id2class.pkl")
    test_args["num_classes"] = len(index2label)

    # Define the same model
    model = AdvSeqLabel(test_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, "./save/saved_model.pkl")
    print("model loaded!")

    # Tester
    tester = SeqLabelTester(test_args)

    # Start testing
    tester.test(model)

    # print test results
    print(tester.show_matrices())
    print("model tested!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run a chinese word segmentation model')
    parser.add_argument('--mode', help='set the model\'s model', choices=['train', 'test', 'infer'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'infer':
        infer()
    else:
        print('no mode specified for model!')
        parser.print_help()
