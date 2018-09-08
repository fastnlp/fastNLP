import sys

sys.path.append("..")
from fastNLP.fastnlp import FastNLP
from fastNLP.fastnlp import interpret_word_seg_results, interpret_cws_pos_results

PATH_TO_CWS_PICKLE_FILES = "/home/zyfeng/fastNLP/reproduction/chinese_word_segment/save/"
PATH_TO_POS_TAG_PICKLE_FILES = "/home/zyfeng/data/crf_seg/"
PATH_TO_TEXT_CLASSIFICATION_PICKLE_FILES = "/home/zyfeng/data/text_classify/"

def word_seg():
    nlp = FastNLP(model_dir=PATH_TO_CWS_PICKLE_FILES)
    nlp.load("cws_basic_model", config_file="cws.cfg", section_name="POS_test")
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


def text_class():
    nlp = FastNLP("./data_for_tests/")
    nlp.load("text_class_model")
    text = "这是最好的基于深度学习的中文分词系统。"
    result = nlp.run(text)
    print(result)
    print("FastNLP finished!")


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


def pos_tag():
    nlp = FastNLP(model_dir=PATH_TO_POS_TAG_PICKLE_FILES)
    nlp.load("pos_tag_model", config_file="pos_tag.config", section_name="pos_tag_model")
    text = ["这是最好的基于深度学习的中文分词系统。",
            "大王叫我来巡山。",
            "我党多年来致力于改善人民生活水平。"]
    results = nlp.run(text)
    for example in results:
        words, labels = [], []
        for res in example:
            words.append(res[0])
            labels.append(res[1])
        print(interpret_cws_pos_results(words, labels))


def text_classify():
    nlp = FastNLP(model_dir=PATH_TO_TEXT_CLASSIFICATION_PICKLE_FILES)
    nlp.load("text_classify_model", config_file="text_classify.cfg", section_name="model")
    text = [
        "世界物联网大会明日在京召开龙头股启动在即",
        "乌鲁木齐市新增一处城市中心旅游目的地",
        "朱元璋的大明朝真的源于明教吗？——告诉你一个真实的“明教”"]
    results = nlp.run(text)
    print(results)
    """
    ['finance', 'travel', 'history']
    """

if __name__ == "__main__":
    text_classify()
