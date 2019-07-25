"""
api/example.py contains all API examples provided by fastNLP.
It is used as a tutorial for API or a test script since it is difficult to test APIs in travis.

"""
from . import CWS, POS, Parser

text = ['编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。',
        '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
        '那么这款无人机到底有多厉害？']


def chinese_word_segmentation():
    cws = CWS(device='cpu')
    print(cws.predict(text))


def chinese_word_segmentation_test():
    cws = CWS(device='cpu')
    print(cws.test("../../test/data_for_tests/zh_sample.conllx"))


def pos_tagging():
    # 输入已分词序列
    text = [['编者', '按：', '7月', '12日', '，', '英国', '航空', '航天', '系统', '公司', '公布', '了', '该', '公司',
             '研制', '的', '第一款', '高科技', '隐形', '无人机', '雷电之神', '。'],
            ['那么', '这', '款', '无人机', '到底', '有', '多', '厉害', '？']]
    pos = POS(device='cpu')
    print(pos.predict(text))


def pos_tagging_test():
    pos = POS(device='cpu')
    print(pos.test("../../test/data_for_tests/zh_sample.conllx"))


def syntactic_parsing():
    text = [['编者', '按：', '7月', '12日', '，', '英国', '航空', '航天', '系统', '公司', '公布', '了', '该', '公司',
             '研制', '的', '第一款', '高科技', '隐形', '无人机', '雷电之神', '。'],
            ['那么', '这', '款', '无人机', '到底', '有', '多', '厉害', '？']]
    parser = Parser(device='cpu')
    print(parser.predict(text))


def syntactic_parsing_test():
    parser = Parser(device='cpu')
    print(parser.test("../../test/data_for_tests/zh_sample.conllx"))


if __name__ == "__main__":
    # chinese_word_segmentation()
    # chinese_word_segmentation_test()
    # pos_tagging()
    # pos_tagging_test()
    syntactic_parsing()
    # syntactic_parsing_test()
