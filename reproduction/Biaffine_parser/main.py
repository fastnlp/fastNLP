import sys

sys.path.extend(['/home/yfshao/workdir/dev_fastnlp'])

import torch
import argparse

from fastNLP.io.dataset_loader import ConllxDataLoader, add_seg_tag
from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance

parser = argparse.ArgumentParser()
parser.add_argument('--pipe', type=str, default='')
parser.add_argument('--gold_data', type=str, default='')
parser.add_argument('--new_data', type=str)
args = parser.parse_args()

pipe = torch.load(args.pipe)['pipeline']
for p in pipe:
    if p.field_name == 'word_list':
        print(p.field_name)
        p.field_name = 'gold_words'
    elif p.field_name == 'pos_list':
        print(p.field_name)
        p.field_name = 'gold_pos'


data = ConllxDataLoader().load(args.gold_data)
ds = DataSet()
for ins1, ins2 in zip(add_seg_tag(data), data):
    ds.append(Instance(words=ins1[0], tag=ins1[1],
                       gold_words=ins2[0], gold_pos=ins2[1],
                       gold_heads=ins2[2], gold_head_tags=ins2[3]))

ds = pipe(ds)

seg_threshold = 0.
pos_threshold = 0.
parse_threshold = 0.74


def get_heads(ins, head_f, word_f):
    head_pred = []
    for i, idx in enumerate(ins[head_f]):
        j = idx - 1 if idx != 0 else i
        head_pred.append(ins[word_f][j])
    return head_pred

def evaluate(ins):
    seg_count = sum([1 for i, j in zip(ins['word_list'], ins['gold_words']) if i == j])
    pos_count = sum([1 for i, j in zip(ins['pos_list'], ins['gold_pos']) if i == j])
    head_count = sum([1 for i, j in zip(ins['heads'], ins['gold_heads']) if i == j])
    total = len(ins['gold_words'])
    return seg_count / total, pos_count / total, head_count / total

def is_ok(x):
    seg, pos, head = x[1]
    return seg > seg_threshold and pos > pos_threshold and head > parse_threshold

res_list = []

for i, ins in enumerate(ds):
    res_list.append((i, evaluate(ins)))

res_list = list(filter(is_ok, res_list))
print('{} {}'.format(len(ds), len(res_list)))

seg_cor, pos_cor, head_cor, label_cor, total = 0,0,0,0,0
for i, _ in res_list:
    ins = ds[i]
    # print(i)
    # print('gold_words:\t', ins['gold_words'])
    # print('predict_words:\t', ins['word_list'])
    # print('gold_tag:\t', ins['gold_pos'])
    # print('predict_tag:\t', ins['pos_list'])
    # print('gold_heads:\t', ins['gold_heads'])
    # print('predict_heads:\t', ins['heads'].tolist())
    # print('gold_head_tags:\t', ins['gold_head_tags'])
    # print('predict_labels:\t', ins['labels'])
    # print()

    head_pred = ins['heads']
    head_gold = ins['gold_heads']
    label_pred = ins['labels']
    label_gold = ins['gold_head_tags']
    total += len(head_gold)
    seg_cor += sum([1 for i, j in zip(ins['word_list'], ins['gold_words']) if i == j])
    pos_cor += sum([1 for i, j in zip(ins['pos_list'], ins['gold_pos']) if i == j])
    length = len(head_gold)
    for i in range(length):
        head_cor += 1 if head_pred[i] == head_gold[i] else 0
        label_cor += 1 if head_pred[i] == head_gold[i] and label_gold[i] == label_pred[i] else 0


print('SEG: {}, POS: {}, UAS: {}, LAS: {}'.format(seg_cor/total, pos_cor/total, head_cor/total, label_cor/total))

colln_path = args.gold_data
new_colln_path = args.new_data

index_list = [x[0] for x in res_list]

with open(colln_path, 'r', encoding='utf-8') as f1, \
        open(new_colln_path, 'w', encoding='utf-8') as f2:
    for idx, ins in enumerate(ds):
        if idx in index_list:
            length = len(ins['gold_words'])
            pad = ['_' for _ in range(length)]
            for x in zip(
                map(str, range(1, length+1)), ins['gold_words'], ins['gold_words'], ins['gold_pos'],
                pad, pad, map(str, ins['gold_heads']), ins['gold_head_tags']):
                new_lines = '\t'.join(x)
                f2.write(new_lines)
                f2.write('\n')
            f2.write('\n')
