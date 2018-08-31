import sys

sys.path.append("..")
from fastNLP.fastnlp import FastNLP

PATH_TO_CWS_PICKLE_FILES = "/home/zyfeng/fastNLP/reproduction/chinese_word_segment/save/"

def word_seg():
    nlp = FastNLP(model_dir=PATH_TO_CWS_PICKLE_FILES)
    nlp.load("cws_basic_model", config_file="cws.cfg", section_name="POS_test")
    text = "这是最好的基于深度学习的中文分词系统。"
    result = nlp.run(text)
    print(result)
    print("FastNLP finished!")


def text_class():
    nlp = FastNLP("./data_for_tests/")
    nlp.load("text_class_model")
    text = "这是最好的基于深度学习的中文分词系统。"
    result = nlp.run(text)
    print(result)
    print("FastNLP finished!")


if __name__ == "__main__":
    word_seg()
