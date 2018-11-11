from fastNLP.api.api import API
from fastNLP.core.dataset import DataSet
from fastNLP.core.predictor import Predictor
from fastNLP.api.pipeline import Pipeline
from fastNLP.api.processor import *
from fastNLP.models.biaffine_parser import BiaffineParser

import torch


class DependencyParser(API):
    def __init__(self):
        super(DependencyParser, self).__init__()

    def predict(self, data):
        self.load('xxx')

        dataset = DataSet()
        dataset = self.pipeline.process(dataset)

        pred = Predictor()
        res = pred.predict(self.model, dataset)
        heads, head_tags = [], []
        for batch in res:
            heads.append(batch['heads'])
            head_tags.append(batch['labels'])
        heads, head_tags = torch.cat(heads, dim=0), torch.cat(head_tags, dim=0)
        return heads, head_tags


    def build(self):
        BOS = '<BOS>'
        NUM = '<NUM>'
        model_args = {}
        load_path = ''
        word_vocab = load(f'{load_path}/word_v.pkl')
        pos_vocab = load(f'{load_path}/pos_v.pkl')
        word_seq = 'word_seq'
        pos_seq = 'pos_seq'

        pipe = Pipeline()
        # build pipeline
        pipe.add_processor(Num2TagProcessor(NUM, 'raw_sentence', word_seq))
        pipe.add_processor(MapFieldProcessor(lambda x: [BOS] + x, word_seq, None))
        pipe.add_processor(MapFieldProcessor(lambda x: [BOS] + x, pos_seq, None))
        pipe.add_processor(IndexerProcessor(word_vocab, word_seq, word_seq+'_idx'))
        pipe.add_processor(IndexerProcessor(pos_vocab, pos_seq, pos_seq+'_idx'))
        pipe.add_processor(MapFieldProcessor(lambda x: len(x), word_seq, 'seq_len'))


        # load model parameters
        self.model = BiaffineParser(**model_args)
        self.pipeline = pipe

