from fastNLP.fastnlp import FastNLP


def foo():
    nlp = FastNLP("./data_for_tests/")
    nlp.load("zh_pos_tag_model")
    text = "这是最好的基于深度学习的中文分词系统。"
    result = nlp.run(text)
    print(result)
    print("FastNLP finished!")


if __name__ == "__main__":
    foo()
