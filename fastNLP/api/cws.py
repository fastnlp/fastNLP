

from fastNLP.api.api import API
from fastNLP.core.dataset import DataSet

class CWS(API):
    def __init__(self, model_path='xxx'):
        super(CWS, self).__init__()
        self.load(model_path)

    def predict(self, sentence, pretrain=False):

        if hasattr(self, 'model') and hasattr(self, 'pipeline'):
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

        # 4. TODO 这里应该要交给一个iterator一样的东西预测这个结果

        # 5. TODO 得到结果，需要考虑是否需要反转回去, 及post_process的操作
        