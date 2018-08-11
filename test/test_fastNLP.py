from fastNLP.fastnlp import FastNLP


def word_seg():
    nlp = FastNLP("./data_for_tests/")
    nlp.load("seq_label_model")
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
    text_class()
