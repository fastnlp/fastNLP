import os
from os.path import exists

def get_data_path(mode, label_type):
    paths = {}
    if mode == 'train':
        paths['train'] = 'data/' + label_type + '/bert.train.jsonl'
        paths['val'] = 'data/' + label_type + '/bert.val.jsonl'
    else:
        paths['test'] = 'data/' + label_type + '/bert.test.jsonl'
    return paths

def get_rouge_path(label_type):
    if label_type == 'others':
        data_path = 'data/' + label_type + '/bert.test.jsonl'
    else:
        data_path = 'data/' + label_type + '/test.jsonl'
    dec_path = 'dec'
    ref_path = 'ref'
    if not exists(ref_path):
        os.makedirs(ref_path)
    if not exists(dec_path):
        os.makedirs(dec_path)
    return data_path, dec_path, ref_path
