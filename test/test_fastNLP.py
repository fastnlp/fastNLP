import sys
sys.path.append("..")
from fastNLP.fastnlp import FastNLP
from fastNLP.fastnlp import interpret_word_seg_results

PATH_TO_CWS_PICKLE_FILES = "/home/zyfeng/fastNLP/reproduction/chinese_word_segment/save/"

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


if __name__ == "__main__":
    word_seg()
