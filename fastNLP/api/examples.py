"""
api/example.py contains all API examples provided by fastNLP.
It is used as a tutorial for API or a test script since it is difficult to test APIs in travis.

"""
from fastNLP.api import CWS, POS

text = ['编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。',
        '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
        '那么这款无人机到底有多厉害？']


def chinese_word_segmentation():
    cws = CWS(device='cpu')
    print(cws.predict(text))


def pos_tagging():
    pos = POS(device='cpu')
    print(pos.predict(text))


if __name__ == "__main__":
    pos_tagging()
