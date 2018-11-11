from fastNLP.api.api import API
from fastNLP.core.dataset import DataSet
from fastNLP.core.predictor import Predictor
from fastNLP.api.pipeline import Pipeline
from fastNLP.api.processor import *
from fastNLP.models.biaffine_parser import BiaffineParser

from fastNLP.core.instance import Instance

import torch


class DependencyParser(API):
    def __init__(self):
        super(DependencyParser, self).__init__()

    def predict(self, data):
        if self.pipeline is None:
            self.pipeline = torch.load('xxx')

        dataset = DataSet()
        for sent, pos_seq in data:
            dataset.append(Instance(sentence=sent, sent_pos=pos_seq))
        dataset = self.pipeline.process(dataset)

        return dataset['heads'], dataset['labels']

if __name__ == '__main__':
    data = [
        (['我', '是', '谁'], ['NR', 'VV', 'NR']),
        (['自古', '英雄', '识', '英雄'], ['AD', 'NN', 'VV', 'NN']),
    ]
    parser = DependencyParser()
    with open('/home/yfshao/workdir/dev_fastnlp/reproduction/Biaffine_parser/pipe/pipeline.pkl', 'rb') as f:
        parser.pipeline = torch.load(f)
    output = parser.predict(data)
    print(output)
