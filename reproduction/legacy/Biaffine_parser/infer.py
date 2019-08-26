import os
import sys

sys.path.extend(['/home/yfshao/workdir/dev_fastnlp'])

from fastNLP.api.processor import *
from fastNLP.models.biaffine_parser import BiaffineParser
from fastNLP.io.config_io import ConfigSection, ConfigLoader

import _pickle as pickle
import torch

def _load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def _load_all(src):
    model_path = src
    src = os.path.dirname(src)

    word_v = _load(src+'/word_v.pkl')
    pos_v = _load(src+'/pos_v.pkl')
    tag_v = _load(src+'/tag_v.pkl')
    pos_pp = torch.load(src+'/pos_pp.pkl')['pipeline']

    model_args = ConfigSection()
    ConfigLoader.load_config('cfg.cfg', {'model': model_args})
    model_args['word_vocab_size'] = len(word_v)
    model_args['pos_vocab_size'] = len(pos_v)
    model_args['num_label'] = len(tag_v)

    model = BiaffineParser(**model_args.data)
    model.load_state_dict(torch.load(model_path))
    return {
        'word_v': word_v,
        'pos_v': pos_v,
        'tag_v': tag_v,
        'model': model,
        'pos_pp':pos_pp,
    }

def build(load_path, save_path):
    BOS = '<BOS>'
    NUM = '<NUM>'
    _dict = _load_all(load_path)
    word_vocab = _dict['word_v']
    pos_vocab = _dict['pos_v']
    tag_vocab = _dict['tag_v']
    pos_pp = _dict['pos_pp']
    model = _dict['model']
    print('load model from {}'.format(load_path))
    word_seq = 'raw_word_seq'
    pos_seq = 'raw_pos_seq'

    # build pipeline
    # input
    pipe = pos_pp
    pipe.pipeline.pop(-1)
    pipe.add_processor(Num2TagProcessor(NUM, 'word_list', word_seq))
    pipe.add_processor(PreAppendProcessor(BOS, word_seq))
    pipe.add_processor(PreAppendProcessor(BOS, 'pos_list', pos_seq))
    pipe.add_processor(IndexerProcessor(word_vocab, word_seq, 'word_seq'))
    pipe.add_processor(IndexerProcessor(pos_vocab, pos_seq, 'pos_seq'))
    pipe.add_processor(SeqLenProcessor('word_seq', 'word_seq_origin_len'))
    pipe.add_processor(SetTensorProcessor({'word_seq':True, 'pos_seq':True, 'word_seq_origin_len':True}, default=False))
    pipe.add_processor(ModelProcessor(model, 'word_seq_origin_len'))
    pipe.add_processor(SliceProcessor(1, None, None, 'head_pred', 'heads'))
    pipe.add_processor(SliceProcessor(1, None, None, 'label_pred', 'label_pred'))
    pipe.add_processor(Index2WordProcessor(tag_vocab, 'label_pred', 'labels'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path+'/pipeline.pkl', 'wb') as f:
        torch.save({'pipeline': pipe}, f)
    print('save pipeline in {}'.format(save_path))


import argparse
parser = argparse.ArgumentParser(description='build pipeline for parser.')
parser.add_argument('--src', type=str, default='/home/yfshao/workdir/dev_fastnlp/reproduction/Biaffine_parser/save')
parser.add_argument('--dst', type=str, default='/home/yfshao/workdir/dev_fastnlp/reproduction/Biaffine_parser/pipe')
args = parser.parse_args()
build(args.src, args.dst)
