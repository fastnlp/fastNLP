
import torch

from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.core.predictor import Predictor


class API:
    def __init__(self):
        self.pipeline = None
        self.model = None

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, name):
        _dict = torch.load(name)
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

        data = self.pipeline(data)

        predictor = Predictor()
        outputs = predictor.predict(self.model, data)

        answers = []
        for out in outputs:
            out = out.numpy()
            for sent in out:
                answers.append([self.tag_vocab.to_word(tag) for tag in sent])
        return answers

    def load(self, name):
        _dict = torch.load(name)
        self.pipeline = _dict['pipeline']
        self.model = _dict['model']
        self.tag_vocab = _dict["tag_vocab"]



class CWS(API):
    def __init__(self, model_path='xxx'):
        super(CWS, self).__init__()
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
