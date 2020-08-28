from utils_ import get_skip_path_trivial, Trie, get_skip_path
from load_data import load_yangjie_rich_pretrain_word_list, load_ontonotes4ner, equip_chinese_ner_with_skip
from pathes import *
from functools import partial
from fastNLP import cache_results
from fastNLP.embeddings.static_embedding import StaticEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.core.metrics import _bmes_tag_to_spans,_bmeso_tag_to_spans
from load_data import load_resume_ner


# embed = StaticEmbedding(None,embedding_dim=2)
# datasets,vocabs,embeddings = load_ontonotes4ner(ontonote4ner_cn_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
#                                                 _refresh=True,index_token=False)
#
# w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
#                                               _refresh=False)
#
# datasets,vocabs,embeddings = equip_chinese_ner_with_skip(datasets,vocabs,embeddings,w_list,yangjie_rich_pretrain_word_path,
#                                                          _refresh=True)
#

def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    # print('in:{}.out:{}'.format(input_string, output_string))
    return output_string





def get_yangjie_bmeso(label_list):
    def get_ner_BMESO_yj(label_list):
        # list_len = len(word_list)
        # assert(list_len == len(label_list)), "word list size unmatch with label list"
        list_len = len(label_list)
        begin_label = 'b-'
        end_label = 'e-'
        single_label = 's-'
        whole_tag = ''
        index_tag = ''
        tag_list = []
        stand_matrix = []
        for i in range(0, list_len):
            # wordlabel = word_list[i]
            current_label = label_list[i].lower()
            if begin_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

            elif single_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
                tag_list.append(whole_tag)
                whole_tag = ""
                index_tag = ""
            elif end_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i))
                whole_tag = ''
                index_tag = ''
            else:
                continue
        if (whole_tag != '') & (index_tag != ''):
            tag_list.append(whole_tag)
        tag_list_len = len(tag_list)

        for i in range(0, tag_list_len):
            if len(tag_list[i]) > 0:
                tag_list[i] = tag_list[i] + ']'
                insert_list = reverse_style(tag_list[i])
                stand_matrix.append(insert_list)
        # print stand_matrix
        return stand_matrix

    def transform_YJ_to_fastNLP(span):
        span = span[1:]
        span_split = span.split(']')
        # print('span_list:{}'.format(span_split))
        span_type = span_split[1]
        # print('span_split[0].split(','):{}'.format(span_split[0].split(',')))
        if ',' in span_split[0]:
            b, e = span_split[0].split(',')
        else:
            b = span_split[0]
            e = b

        b = int(b)
        e = int(e)

        e += 1

        return (span_type, (b, e))
    yj_form = get_ner_BMESO_yj(label_list)
    # print('label_list:{}'.format(label_list))
    # print('yj_from:{}'.format(yj_form))
    fastNLP_form = list(map(transform_YJ_to_fastNLP,yj_form))
    return fastNLP_form


# tag_list = ['O', 'B-singer', 'M-singer', 'E-singer', 'O', 'O']
# span_list = get_ner_BMES(tag_list)
# print(span_list)
# yangjie_label_list = ['B-NAME', 'E-NAME', 'O', 'B-CONT', 'M-CONT', 'E-CONT', 'B-RACE', 'E-RACE', 'B-TITLE', 'M-TITLE', 'E-TITLE', 'B-EDU', 'M-EDU', 'E-EDU', 'B-ORG', 'M-ORG', 'E-ORG', 'M-NAME', 'B-PRO', 'M-PRO', 'E-PRO', 'S-RACE', 'S-NAME', 'B-LOC', 'M-LOC', 'E-LOC', 'M-RACE', 'S-ORG']
# my_label_list = ['O', 'M-ORG', 'M-TITLE', 'B-TITLE', 'E-TITLE', 'B-ORG', 'E-ORG', 'M-EDU', 'B-NAME', 'E-NAME', 'B-EDU', 'E-EDU', 'M-NAME', 'M-PRO', 'M-CONT', 'B-PRO', 'E-PRO', 'B-CONT', 'E-CONT', 'M-LOC', 'B-RACE', 'E-RACE', 'S-NAME', 'B-LOC', 'E-LOC', 'M-RACE', 'S-RACE', 'S-ORG']
# yangjie_label = set(yangjie_label_list)
# my_label = set(my_label_list)

a = torch.tensor([0,2,0,3])
b = (a==0)
print(b)
print(b.float())
from fastNLP import RandomSampler

# f = open('/remote-home/xnli/weight_debug/lattice_yangjie.pkl','rb')
# weight_dict = torch.load(f)
# print(weight_dict.keys())
# for k,v in weight_dict.items():
#     print("{}:{}".format(k,v.size()))