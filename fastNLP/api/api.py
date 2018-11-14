import warnings

import torch

warnings.filterwarnings('ignore')
import os

from fastNLP.core.dataset import DataSet
from fastNLP.api.model_zoo import load_url
from fastNLP.api.processor import ModelProcessor
from reproduction.chinese_word_segment.cws_io.cws_reader import ConlluCWSReader
from reproduction.pos_tag_model.pos_io.pos_reader import ConlluPOSReader
from reproduction.Biaffine_parser.util import ConllxDataLoader, add_seg_tag
from fastNLP.core.instance import Instance
from fastNLP.core.sampler import SequentialSampler
from fastNLP.core.batch import Batch
from reproduction.chinese_word_segment.utils import calculate_pre_rec_f1
from fastNLP.api.pipeline import Pipeline
from fastNLP.core.metrics import SeqLabelEvaluator2
from fastNLP.core.tester import Tester

model_urls = {
}


class API:
    def __init__(self):
        self.pipeline = None

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, path, device):
        if os.path.exists(os.path.expanduser(path)):
            _dict = torch.load(path, map_location='cpu')
        else:
            _dict = load_url(path, map_location='cpu')
        self.pipeline = _dict['pipeline']
        self._dict = _dict
        for processor in self.pipeline.pipeline:
            if isinstance(processor, ModelProcessor):
                processor.set_model_device(device)


class POS(API):
    """FastNLP API for Part-Of-Speech tagging.

    """

    def __init__(self, model_path=None, device='cpu'):
        super(POS, self).__init__()
        if model_path is None:
            model_path = model_urls['pos']

        self.load(model_path, device)

    def predict(self, content):
        """

        :param content: list of list of str. Each string is a token(word).
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

    def test(self, filepath):

        tag_proc = self._dict['tag_indexer']

        model = self.pipeline.pipeline[2].model
        pipeline = self.pipeline.pipeline[0:2]
        pipeline.append(tag_proc)
        pp = Pipeline(pipeline)

        reader = ConlluPOSReader()
        te_dataset = reader.load(filepath)

        evaluator = SeqLabelEvaluator2('word_seq_origin_len')
        end_tagidx_set = set()
        tag_proc.vocab.build_vocab()
        for key, value in tag_proc.vocab.word2idx.items():
            if key.startswith('E-'):
                end_tagidx_set.add(value)
            if key.startswith('S-'):
                end_tagidx_set.add(value)
        evaluator.end_tagidx_set = end_tagidx_set

        default_valid_args = {"batch_size": 64,
                              "use_cuda": True, "evaluator": evaluator}

        pp(te_dataset)
        te_dataset.set_is_target(truth=True)

        tester = Tester(**default_valid_args)

        test_result = tester.test(model, te_dataset)

        f1 = round(test_result['F'] * 100, 2)
        pre = round(test_result['P'] * 100, 2)
        rec = round(test_result['R'] * 100, 2)
        print("f1:{:.2f}, pre:{:.2f}, rec:{:.2f}".format(f1, pre, rec))

        return f1, pre, rec


class CWS(API):
    def __init__(self, model_path=None, device='cpu'):
        super(CWS, self).__init__()
        if model_path is None:
            model_path = model_urls['cws']

        self.load(model_path, device)

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

    def test(self, filepath):

        tag_proc = self._dict['tag_indexer']
        cws_model = self.pipeline.pipeline[-2].model
        pipeline = self.pipeline.pipeline[:5]

        pipeline.insert(1, tag_proc)
        pp = Pipeline(pipeline)

        reader = ConlluCWSReader()

        # te_filename = '/home/hyan/ctb3/test.conllx'
        te_dataset = reader.load(filepath)
        pp(te_dataset)

        batch_size = 64
        te_batcher = Batch(te_dataset, batch_size, SequentialSampler(), use_cuda=False)
        pre, rec, f1 = calculate_pre_rec_f1(cws_model, te_batcher, type='bmes')
        f1 = round(f1 * 100, 2)
        pre = round(pre * 100, 2)
        rec = round(rec * 100, 2)
        print("f1:{:.2f}, pre:{:.2f}, rec:{:.2f}".format(f1, pre, rec))

        return f1, pre, rec


class Analyzer:
    def __init__(self, seg=True, pos=True, parser=True, device='cpu'):

        self.seg = seg
        self.pos = pos
        self.parser = parser

        if self.seg:
            self.cws = CWS(device=device)
        if self.pos:
            self.pos = POS(device=device)
        if parser:
            self.parser = None

    def predict(self, content):
        output_dict = {}
        if self.seg:
            seg_output = self.cws.predict(content)
            output_dict['seg'] = seg_output
        if self.pos:
            pos_output = self.pos.predict(content)
            output_dict['pos'] = pos_output
        if self.parser:
            parser_output = self.parser.predict(content)
            output_dict['parser'] = parser_output

        return output_dict

    def test(self, filepath):
        output_dict = {}
        if self.seg:
            seg_output = self.cws.test(filepath)
            output_dict['seg'] = seg_output
        if self.pos:
            pos_output = self.pos.test(filepath)
            output_dict['pos'] = pos_output
        if self.parser:
            parser_output = self.parser.test(filepath)
            output_dict['parser'] = parser_output

        return output_dict


if __name__ == "__main__":
    # pos_model_path = '../../reproduction/pos_tag_model/pos_crf.pkl'
    # pos = POS(device='cpu')
    # s = ['编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。' ,
    #     '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
    #      '那么这款无人机到底有多厉害？']
    # print(pos.test('/Users/yh/Desktop/test_data/small_test.conll'))
    # print(pos.predict(s))

    # cws_model_path = '../../reproduction/chinese_word_segment/models/cws_crf.pkl'
    cws = CWS(device='cpu')
    s = ['本品是一个抗酸抗胆汁的胃黏膜保护剂' ,
        '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
         '那么这款无人机到底有多厉害？']
    print(cws.test('/Users/yh/Desktop/test_data/small_test.conll'))
    print(cws.predict(s))


