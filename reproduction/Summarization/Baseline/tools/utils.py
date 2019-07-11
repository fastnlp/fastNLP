#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import os
import shutil
import copy
import datetime
import numpy as np

from rouge import Rouge

from .logger import *
# from data import *

import sys
sys.setrecursionlimit(10000)

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}", 
        "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'} 

def clean(x): 
    return re.sub( 
            r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''", 
            lambda m: REMAP.get(m.group()), x) 


def rouge_eval(hyps, refer):
    rouge = Rouge()
    # print(hyps)
    # print(refer)
    # print(rouge.get_scores(hyps, refer))
    try:
        score = rouge.get_scores(hyps, refer)[0]
        mean_score = np.mean([score["rouge-1"]["f"], score["rouge-2"]["f"], score["rouge-l"]["f"]])
    except:
        mean_score = 0.0
    return mean_score

def rouge_all(hyps, refer):
    rouge = Rouge()
    score = rouge.get_scores(hyps, refer)[0]
    # mean_score = np.mean([score["rouge-1"]["f"], score["rouge-2"]["f"], score["rouge-l"]["f"]])
    return score

def eval_label(match_true, pred, true, total, match):
    match_true, pred, true, match = match_true.float(), pred.float(), true.float(), match.float()
    try:
        accu = match / total
        precision = match_true / pred
        recall = match_true / true
        F = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        F = 0.0
        logger.error("[Error] float division by zero")
    return accu, precision, recall, F


def pyrouge_score(hyps, refer, remap = True):
    from pyrouge import Rouge155
    nowTime=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    PYROUGE_ROOT = os.path.join('/remote-home/dqwang/', nowTime)
    SYSTEM_PATH = os.path.join(PYROUGE_ROOT,'gold')
    MODEL_PATH = os.path.join(PYROUGE_ROOT,'system')
    if os.path.exists(SYSTEM_PATH):
        shutil.rmtree(SYSTEM_PATH)
    os.makedirs(SYSTEM_PATH)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    os.makedirs(MODEL_PATH)

    if remap == True:
        refer = clean(refer)
        hyps = clean(hyps)

    system_file = os.path.join(SYSTEM_PATH, 'Reference.0.txt')
    model_file = os.path.join(MODEL_PATH, 'Model.A.0.txt')
    with open(system_file, 'wb') as f:
        f.write(refer.encode('utf-8'))
    with open(model_file, 'wb') as f:
        f.write(hyps.encode('utf-8'))

    r = Rouge155('/home/dqwang/ROUGE/RELEASE-1.5.5')

    r.system_dir = SYSTEM_PATH
    r.model_dir = MODEL_PATH
    r.system_filename_pattern = 'Reference.(\d+).txt'
    r.model_filename_pattern = 'Model.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate(rouge_args="-e /home/dqwang/ROUGE/RELEASE-1.5.5/data -a -m -n 2 -d")
    output_dict = r.output_to_dict(output)

    shutil.rmtree(PYROUGE_ROOT)

    scores = {}
    scores['rouge-1'], scores['rouge-2'], scores['rouge-l'] = {}, {}, {}
    scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'] = output_dict['rouge_1_precision'], output_dict['rouge_1_recall'], output_dict['rouge_1_f_score']
    scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'] = output_dict['rouge_2_precision'], output_dict['rouge_2_recall'], output_dict['rouge_2_f_score']
    scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'] = output_dict['rouge_l_precision'], output_dict['rouge_l_recall'], output_dict['rouge_l_f_score']
    return scores
    
def pyrouge_score_all(hyps_list, refer_list, remap = True):
    from pyrouge import Rouge155
    nowTime=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    PYROUGE_ROOT = os.path.join('/remote-home/dqwang/', nowTime)
    SYSTEM_PATH = os.path.join(PYROUGE_ROOT,'gold')
    MODEL_PATH = os.path.join(PYROUGE_ROOT,'system')
    if os.path.exists(SYSTEM_PATH):
        shutil.rmtree(SYSTEM_PATH)
    os.makedirs(SYSTEM_PATH)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    os.makedirs(MODEL_PATH)

    assert len(hyps_list) == len(refer_list)
    for i in range(len(hyps_list)):
        system_file = os.path.join(SYSTEM_PATH, 'Reference.%d.txt' % i)
        model_file = os.path.join(MODEL_PATH, 'Model.A.%d.txt' % i)

        refer = clean(refer_list[i]) if remap else refer_list[i]
        hyps = clean(hyps_list[i]) if remap else hyps_list[i]

        with open(system_file, 'wb') as f:
            f.write(refer.encode('utf-8'))
        with open(model_file, 'wb') as f:
            f.write(hyps.encode('utf-8'))

    r = Rouge155('/remote-home/dqwang/ROUGE/RELEASE-1.5.5')

    r.system_dir = SYSTEM_PATH
    r.model_dir = MODEL_PATH
    r.system_filename_pattern = 'Reference.(\d+).txt'
    r.model_filename_pattern = 'Model.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate(rouge_args="-e /remote-home/dqwang/ROUGE/RELEASE-1.5.5/data -a -m -n 2 -d")
    output_dict = r.output_to_dict(output)

    shutil.rmtree(PYROUGE_ROOT)

    scores = {}
    scores['rouge-1'], scores['rouge-2'], scores['rouge-l'] = {}, {}, {}
    scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'] = output_dict['rouge_1_precision'], output_dict['rouge_1_recall'], output_dict['rouge_1_f_score']
    scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'] = output_dict['rouge_2_precision'], output_dict['rouge_2_recall'], output_dict['rouge_2_f_score']
    scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'] = output_dict['rouge_l_precision'], output_dict['rouge_l_recall'], output_dict['rouge_l_f_score']
    return scores


def pyrouge_score_all_multi(hyps_list, refer_list, remap = True):
    from pyrouge import Rouge155
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    PYROUGE_ROOT = os.path.join('/remote-home/dqwang/', nowTime)
    SYSTEM_PATH = os.path.join(PYROUGE_ROOT, 'system')
    MODEL_PATH = os.path.join(PYROUGE_ROOT, 'gold')
    if os.path.exists(SYSTEM_PATH):
        shutil.rmtree(SYSTEM_PATH)
    os.makedirs(SYSTEM_PATH)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    os.makedirs(MODEL_PATH)

    assert len(hyps_list) == len(refer_list)
    for i in range(len(hyps_list)):
        system_file = os.path.join(SYSTEM_PATH, 'Model.%d.txt' % i)
        # model_file = os.path.join(MODEL_PATH, 'Reference.A.%d.txt' % i)

        hyps = clean(hyps_list[i]) if remap else hyps_list[i]
        
        with open(system_file, 'wb') as f:
            f.write(hyps.encode('utf-8'))

        referType = ["A", "B", "C", "D", "E", "F", "G"]
        for j in range(len(refer_list[i])):
            model_file = os.path.join(MODEL_PATH, "Reference.%s.%d.txt" % (referType[j], i))
            refer = clean(refer_list[i][j]) if remap else refer_list[i][j]
            with open(model_file, 'wb') as f:
                f.write(refer.encode('utf-8'))

    r = Rouge155('/remote-home/dqwang/ROUGE/RELEASE-1.5.5')

    r.system_dir = SYSTEM_PATH
    r.model_dir = MODEL_PATH
    r.system_filename_pattern = 'Model.(\d+).txt'
    r.model_filename_pattern = 'Reference.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate(rouge_args="-e /remote-home/dqwang/ROUGE/RELEASE-1.5.5/data -a -m -n 2 -d")
    output_dict = r.output_to_dict(output)

    shutil.rmtree(PYROUGE_ROOT)

    scores = {}
    scores['rouge-1'], scores['rouge-2'], scores['rouge-l'] = {}, {}, {}
    scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'] = output_dict['rouge_1_precision'], output_dict['rouge_1_recall'], output_dict['rouge_1_f_score']
    scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'] = output_dict['rouge_2_precision'], output_dict['rouge_2_recall'], output_dict['rouge_2_f_score']
    scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'] = output_dict['rouge_l_precision'], output_dict['rouge_l_recall'], output_dict['rouge_l_f_score']
    return scores

def cal_label(article, abstract):
    hyps_list = article

    refer = abstract
    scores = []
    for hyps in hyps_list:
        mean_score = rouge_eval(hyps, refer)
        scores.append(mean_score)

    selected = []
    selected.append(int(np.argmax(scores)))
    selected_sent_cnt = 1

    best_rouge = np.max(scores)
    while selected_sent_cnt < len(hyps_list):
        cur_max_rouge = 0.0
        cur_max_idx = -1
        for i in range(len(hyps_list)):
            if i not in selected:
                temp = copy.deepcopy(selected)
                temp.append(i)
                hyps = "\n".join([hyps_list[idx] for idx in np.sort(temp)])
                cur_rouge = rouge_eval(hyps, refer)
                if cur_rouge > cur_max_rouge:
                    cur_max_rouge = cur_rouge
                    cur_max_idx = i
        if cur_max_rouge != 0.0 and cur_max_rouge >= best_rouge:
            selected.append(cur_max_idx)
            selected_sent_cnt += 1
            best_rouge = cur_max_rouge
        else:
            break

    # label = np.zeros(len(hyps_list), dtype=int)
    # label[np.array(selected)] = 1
    # return list(label)
    return selected

def cal_label_limited3(article, abstract):
    hyps_list = article

    refer = abstract
    scores = []
    for hyps in hyps_list:
        try:
            mean_score = rouge_eval(hyps, refer)
            scores.append(mean_score)
        except ValueError:
            scores.append(0.0)

    selected = []
    selected.append(np.argmax(scores))
    selected_sent_cnt = 1

    best_rouge = np.max(scores)
    while selected_sent_cnt < len(hyps_list) and selected_sent_cnt < 3:
        cur_max_rouge = 0.0
        cur_max_idx = -1
        for i in range(len(hyps_list)):
            if i not in selected:
                temp = copy.deepcopy(selected)
                temp.append(i)
                hyps = "\n".join([hyps_list[idx] for idx in np.sort(temp)])
                cur_rouge = rouge_eval(hyps, refer)
                if cur_rouge > cur_max_rouge:
                    cur_max_rouge = cur_rouge
                    cur_max_idx = i
        selected.append(cur_max_idx)
        selected_sent_cnt += 1
        best_rouge = cur_max_rouge
    
    # logger.info(selected)
    # label = np.zeros(len(hyps_list), dtype=int)
    # label[np.array(selected)] = 1
    # return list(label)
    return selected
    
import torch
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:]).contiguous()
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0.0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_non_pad_mask(seq):

    assert seq.dim() == 2

    return seq.ne(0.0).type(torch.float).unsqueeze(-1)
