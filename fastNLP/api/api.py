import torch
import warnings
warnings.filterwarnings('ignore')
import os

from fastNLP.core.dataset import DataSet
from fastNLP.api.model_zoo import load_url

model_urls = {
}


class API:
    def __init__(self):
        self.pipeline = None

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, path):
        if os.path.exists(os.path.expanduser(path)):
            _dict = torch.load(path)
        else:
            _dict = load_url(path)
        self.pipeline = _dict['pipeline']


class POS(API):
    """FastNLP API for Part-Of-Speech tagging.

    """

    def __init__(self, model_path=None):
        super(POS, self).__init__()
        if model_path is None:
            model_path = model_urls['pos']

        self.load(model_path)

    def predict(self, content):
        """

        :param query: list of list of str. Each string is a token(word).
        :return answer: list of list of str. Each string is a tag.
        """
        if not hasattr(self, 'pipeline'):
            raise ValueError("You have to load model first.")

        sentence_list = []
        # 1. 检查sentence的类型
        if isinstance(content, str):
            sentence_list.append(content)
        elif isinstance(content, list):
            sentence_list = content

        # 2. 组建dataset
        dataset = DataSet()
        dataset.add_field('words', sentence_list)

        # 3. 使用pipeline
        self.pipeline(dataset)

        output = dataset['word_pos_output'].content
        if isinstance(content, str):
            return output[0]
        elif isinstance(content, list):
            return output


class CWS(API):
    def __init__(self, model_path=None):
        super(CWS, self).__init__()
        if model_path is None:
            model_path = model_urls['cws']

        self.load(model_path)

    def predict(self, content):

        if not hasattr(self, 'pipeline'):
            raise ValueError("You have to load model first.")

        sentence_list = []
        # 1. 检查sentence的类型
        if isinstance(content, str):
            sentence_list.append(content)
        elif isinstance(content, list):
            sentence_list = content

        # 2. 组建dataset
        dataset = DataSet()
        dataset.add_field('raw_sentence', sentence_list)

        # 3. 使用pipeline
        self.pipeline(dataset)

        output = dataset['output'].content
        if isinstance(content, str):
            return output[0]
        elif isinstance(content, list):
            return output


if __name__ == "__main__":
    pos = POS()
    s = ['编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。' ,
        '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
         '那么这款无人机到底有多厉害？']
    print(pos.predict(s))

    # cws = CWS()
    # s = ['编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。' ,
    #     '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
    #      '那么这款无人机到底有多厉害？']
    # print(cws.predict(s))

