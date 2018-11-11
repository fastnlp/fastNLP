import torch

from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance

model_urls = {
    'cws': "",

}


class API:
    def __init__(self):
        self.pipeline = None

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, path):


        _dict = torch.load(path)
        self.pipeline = _dict['pipeline']



class POS_tagger(API):
    """FastNLP API for Part-Of-Speech tagging.

    """

    def __init__(self):
        super(POS_tagger, self).__init__()

    def predict(self, query):
        """

        :param query: list of list of str. Each string is a token(word).
        :return answer: list of list of str. Each string is a tag.
        """
        self.load("/home/zyfeng/fastnlp_0.2.0/reproduction/pos_tag_model/model_pp.pkl")

        data = DataSet()
        for example in query:
            data.append(Instance(words=example))

        out = self.pipeline(data)

        return [x["outputs"] for x in out]

    def load(self, name):
        _dict = torch.load(name)
        self.pipeline = _dict['pipeline']



class CWS(API):
    def __init__(self, model_path=None, pretrain=True):
        super(CWS, self).__init__()
        # 1. 这里修改为检查
        if model_path is None:
            model_path = model_urls['cws']


        self.load(model_path)

    def predict(self, sentence, pretrain=False):

        if hasattr(self, 'pipeline'):
            raise ValueError("You have to load model first. Or specify pretrain=True.")

        sentence_list = []
        # 1. 检查sentence的类型
        if isinstance(sentence, str):
            sentence_list.append(sentence)
        elif isinstance(sentence, list):
            sentence_list = sentence

        # 2. 组建dataset
        dataset = DataSet()
        dataset.add_field('raw_sentence', sentence_list)

        # 3. 使用pipeline
        self.pipeline(dataset)

        output = dataset['output']
        if isinstance(sentence, str):
            return output[0]
        elif isinstance(sentence, list):
            return output


if __name__ == "__main__":
    tagger = POS_tagger()
    print(tagger.predict([["我", "是", "学生", "。"], ["我", "是", "学生", "。"]]))

    from torchvision import models
    models.resnet18()
