import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.loader.dataset_loader import PeopleDailyCorpusLoader, BaseLoader
from fastNLP.core.preprocess import SeqLabelPreprocess, load_pickle
from fastNLP.saver.model_saver import ModelSaver
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.core.tester import SeqLabelTester
from fastNLP.models.sequence_modeling import AdvSeqLabel
from fastNLP.core.predictor import SeqLabelInfer

# not in the file's dir
if len(os.path.dirname(__file__)) != 0:
    os.chdir(os.path.dirname(__file__))
datadir = "/home/zyfeng/data/"
cfgfile = './pos_tag.cfg'
data_name = "CWS_POS_TAG_NER_people_daily.txt"

pos_tag_data_path = os.path.join(datadir, data_name)
pickle_path = "save"
data_infer_path = os.path.join(datadir, "infer.utf8")


def infer():
    # Config Loader
    test_args = ConfigSection()
    ConfigLoader("config").load_config(cfgfile, {"POS_test": test_args})

    # fetch dictionary size and number of labels from pickle files
    word2index = load_pickle(pickle_path, "word2id.pkl")
    test_args["vocab_size"] = len(word2index)
    index2label = load_pickle(pickle_path, "class2id.pkl")
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
    raw_data_loader = BaseLoader(data_infer_path)
    infer_data = raw_data_loader.load_lines()
    print('data loaded')

    # Inference interface
    infer = SeqLabelInfer(pickle_path)
    results = infer.predict(model, infer_data)

    print(results)
    print("Inference finished!")


def train(): 
    # load config
    trainer_args = ConfigSection()
    model_args = ConfigSection()
    ConfigLoader().load_config(cfgfile, {"train": train_args, "test": test_args})

    # Data Loader
    loader = PeopleDailyCorpusLoader()
    train_data, _ = loader.load()

    # TODO: define processors
    
    # define pipeline
    pp = Pipeline()
    # TODO: pp.add_processor()

    # run the pipeline, get data_set
    train_data = pp(train_data)

    # define a model
    model = AdvSeqLabel(train_args)

    # call trainer to train
    trainer = SeqLabelTrainer(train_args)
    trainer.train(model, data_train, data_dev)

    # save model
    ModelSaver("./saved_model.pkl").save_pytorch(model, param_only=False)

    # TODO:save pipeline



def test():
    # Config Loader
    test_args = ConfigSection()
    ConfigLoader("config").load_config(cfgfile, {"POS_test": test_args})

    # fetch dictionary size and number of labels from pickle files
    word2index = load_pickle(pickle_path, "word2id.pkl")
    test_args["vocab_size"] = len(word2index)
    index2label = load_pickle(pickle_path, "class2id.pkl")
    test_args["num_classes"] = len(index2label)

    # load dev data
    dev_data = load_pickle(pickle_path, "data_dev.pkl")

    # Define the same model
    model = AdvSeqLabel(test_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, "./save/saved_model.pkl")
    print("model loaded!")

    # Tester
    tester = SeqLabelTester(**test_args.data)

    # Start testing
    tester.test(model, dev_data)

    # print test results
    print(tester.show_metrics())
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
