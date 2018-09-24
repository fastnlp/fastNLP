# encoding: utf-8
import os

from fastNLP.core.preprocess import save_pickle
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.fastnlp import FastNLP
from fastNLP.fastnlp import interpret_word_seg_results, interpret_cws_pos_results
from fastNLP.models.cnn_text_classification import CNNText
from fastNLP.models.sequence_modeling import AdvSeqLabel
from fastNLP.saver.model_saver import ModelSaver

PATH_TO_CWS_PICKLE_FILES = "/home/zyfeng/fastNLP/reproduction/chinese_word_segment/save/"
PATH_TO_POS_TAG_PICKLE_FILES = "/home/zyfeng/data/crf_seg/"
PATH_TO_TEXT_CLASSIFICATION_PICKLE_FILES = "/home/zyfeng/data/text_classify/"

DEFAULT_PADDING_LABEL = '<pad>'  # dict index = 0
DEFAULT_UNKNOWN_LABEL = '<unk>'  # dict index = 1
DEFAULT_RESERVED_LABEL = ['<reserved-2>',
                          '<reserved-3>',
                          '<reserved-4>']  # dict index = 2~4

DEFAULT_WORD_TO_INDEX = {DEFAULT_PADDING_LABEL: 0, DEFAULT_UNKNOWN_LABEL: 1,
                         DEFAULT_RESERVED_LABEL[0]: 2, DEFAULT_RESERVED_LABEL[1]: 3,
                         DEFAULT_RESERVED_LABEL[2]: 4}


def word_seg(model_dir, config, section):
    nlp = FastNLP(model_dir=model_dir)
    nlp.load("cws_basic_model", config_file=config, section_name=section)
    text = ["这是最好的基于深度学习的中文分词系统。",
            "大王叫我来巡山。",
            "我党多年来致力于改善人民生活水平。"]
    results = nlp.run(text)
    print(results)
    for example in results:
        words, labels = [], []
        for res in example:
            words.append(res[0])
            labels.append(res[1])
        print(interpret_word_seg_results(words, labels))


def mock_cws():
    os.makedirs("mock", exist_ok=True)
    text = ["这是最好的基于深度学习的中文分词系统。",
            "大王叫我来巡山。",
            "我党多年来致力于改善人民生活水平。"]

    word2id = Vocabulary()
    word_list = [ch for ch in "".join(text)]
    word2id.update(word_list)
    save_pickle(word2id, "./mock/", "word2id.pkl")

    class2id = Vocabulary(need_default=False)
    label_list = ['B', 'M', 'E', 'S']
    class2id.update(label_list)
    save_pickle(class2id, "./mock/", "class2id.pkl")

    model_args = {"vocab_size": len(word2id), "word_emb_dim": 50, "rnn_hidden_units": 50, "num_classes": len(class2id)}
    config_file = """
    [test_section]
    vocab_size = {}
    word_emb_dim = 50
    rnn_hidden_units = 50
    num_classes = {}
    """.format(len(word2id), len(class2id))
    with open("mock/test.cfg", "w", encoding="utf-8") as f:
        f.write(config_file)

    model = AdvSeqLabel(model_args)
    ModelSaver("mock/cws_basic_model_v_0.pkl").save_pytorch(model)


def test_word_seg():
    # fake the model and pickles
    print("start mocking")
    mock_cws()
    # run the inference codes
    print("start testing")
    word_seg("./mock/", "test.cfg", "test_section")
    # clean up environments
    print("clean up")
    os.system("rm -rf mock")


def pos_tag(model_dir, config, section):
    nlp = FastNLP(model_dir=model_dir)
    nlp.load("pos_tag_model", config_file=config, section_name=section)
    text = ["这是最好的基于深度学习的中文分词系统。",
            "大王叫我来巡山。",
            "我党多年来致力于改善人民生活水平。"]
    results = nlp.run(text)
    for example in results:
        words, labels = [], []
        for res in example:
            words.append(res[0])
            labels.append(res[1])
        try:
            print(interpret_cws_pos_results(words, labels))
        except RuntimeError:
            print("inconsistent pos tags. this is for test only.")


def mock_pos_tag():
    os.makedirs("mock", exist_ok=True)
    text = ["这是最好的基于深度学习的中文分词系统。",
            "大王叫我来巡山。",
            "我党多年来致力于改善人民生活水平。"]

    vocab = Vocabulary()
    word_list = [ch for ch in "".join(text)]
    vocab.update(word_list)
    save_pickle(vocab, "./mock/", "word2id.pkl")

    idx2label = Vocabulary(need_default=False)
    label_list = ['B-n', 'M-v', 'E-nv', 'S-adj', 'B-v', 'M-vn', 'S-adv']
    idx2label.update(label_list)
    save_pickle(idx2label, "./mock/", "class2id.pkl")

    model_args = {"vocab_size": len(vocab), "word_emb_dim": 50, "rnn_hidden_units": 50, "num_classes": len(idx2label)}
    config_file = """
        [test_section]
        vocab_size = {}
        word_emb_dim = 50
        rnn_hidden_units = 50
        num_classes = {}
        """.format(len(vocab), len(idx2label))
    with open("mock/test.cfg", "w", encoding="utf-8") as f:
        f.write(config_file)

    model = AdvSeqLabel(model_args)
    ModelSaver("mock/pos_tag_model_v_0.pkl").save_pytorch(model)


def test_pos_tag():
    mock_pos_tag()
    pos_tag("./mock/", "test.cfg", "test_section")
    os.system("rm -rf mock")


def text_classify(model_dir, config, section):
    nlp = FastNLP(model_dir=model_dir)
    nlp.load("text_classify_model", config_file=config, section_name=section)
    text = [
        "世界物联网大会明日在京召开龙头股启动在即",
        "乌鲁木齐市新增一处城市中心旅游目的地",
        "朱元璋的大明朝真的源于明教吗？——告诉你一个真实的“明教”"]
    results = nlp.run(text)
    print(results)


def mock_text_classify():
    os.makedirs("mock", exist_ok=True)
    text = ["世界物联网大会明日在京召开龙头股启动在即",
            "乌鲁木齐市新增一处城市中心旅游目的地",
            "朱元璋的大明朝真的源于明教吗？——告诉你一个真实的“明教”"
            ]
    vocab = Vocabulary()
    word_list = [ch for ch in "".join(text)]
    vocab.update(word_list)
    save_pickle(vocab, "./mock/", "word2id.pkl")

    idx2label = Vocabulary(need_default=False)
    label_list = ['class_A', 'class_B', 'class_C', 'class_D', 'class_E', 'class_F']
    idx2label.update(label_list)
    save_pickle(idx2label, "./mock/", "class2id.pkl")

    model_args = {"vocab_size": len(vocab), "word_emb_dim": 50, "rnn_hidden_units": 50, "num_classes": len(idx2label)}
    config_file = """
            [test_section]
            vocab_size = {}
            word_emb_dim = 50
            rnn_hidden_units = 50
            num_classes = {}
            """.format(len(vocab), len(idx2label))
    with open("mock/test.cfg", "w", encoding="utf-8") as f:
        f.write(config_file)

    model = CNNText(model_args)
    ModelSaver("mock/text_class_model_v0.pkl").save_pytorch(model)


def test_text_classify():
    mock_text_classify()
    text_classify("./mock/", "test.cfg", "test_section")
    os.system("rm -rf mock")


def test_word_seg_interpret():
    foo = [[('这', 'S'), ('是', 'S'), ('最', 'S'), ('好', 'S'), ('的', 'S'), ('基', 'B'), ('于', 'E'), ('深', 'B'), ('度', 'E'),
            ('学', 'B'), ('习', 'E'), ('的', 'S'), ('中', 'B'), ('文', 'E'), ('分', 'B'), ('词', 'E'), ('系', 'B'), ('统', 'E'),
            ('。', 'S')]]
    chars = [x[0] for x in foo[0]]
    labels = [x[1] for x in foo[0]]
    print(interpret_word_seg_results(chars, labels))


def test_interpret_cws_pos_results():
    foo = [
        [('这', 'S-r'), ('是', 'S-v'), ('最', 'S-d'), ('好', 'S-a'), ('的', 'S-u'), ('基', 'B-p'), ('于', 'E-p'), ('深', 'B-d'),
         ('度', 'E-d'), ('学', 'B-v'), ('习', 'E-v'), ('的', 'S-u'), ('中', 'B-nz'), ('文', 'E-nz'), ('分', 'B-vn'),
         ('词', 'E-vn'), ('系', 'B-n'), ('统', 'E-n'), ('。', 'S-w')]
    ]
    chars = [x[0] for x in foo[0]]
    labels = [x[1] for x in foo[0]]
    print(interpret_cws_pos_results(chars, labels))

if __name__ == "__main__":
    test_word_seg()
    test_pos_tag()
    test_text_classify()
    test_word_seg_interpret()
    test_interpret_cws_pos_results()
