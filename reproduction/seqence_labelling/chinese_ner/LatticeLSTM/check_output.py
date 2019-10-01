import torch.nn as nn
from pathes import *
from load_data import load_ontonotes4ner,equip_chinese_ner_with_skip,load_yangjie_rich_pretrain_word_list,load_resume_ner
from fastNLP.embeddings import StaticEmbedding
from models import LatticeLSTM_SeqLabel,LSTM_SeqLabel,LatticeLSTM_SeqLabel_V1
from fastNLP import CrossEntropyLoss,SpanFPreRecMetric,Trainer,AccuracyMetric,LossInForward
import torch.optim as optim
import argparse
import torch
import sys
from utils_ import LatticeLexiconPadder,SpanFPreRecMetric_YJ
from fastNLP import Tester
import fitlog
from fastNLP.core.callback import FitlogCallback
from utils import set_seed
import os
from fastNLP import LRScheduler
from torch.optim.lr_scheduler import LambdaLR



# sys.path.append('.')
# sys.path.append('..')
# for p in sys.path:
#     print(p)
# fitlog.add_hyper_in_file (__file__) # record your hyperparameters
########hyper

########hyper

parser = argparse.ArgumentParser()
parser.add_argument('--device',default='cpu')
parser.add_argument('--debug',default=True)

parser.add_argument('--batch',default=1)
parser.add_argument('--test_batch',default=1024)
parser.add_argument('--optim',default='sgd',help='adam|sgd')
parser.add_argument('--lr',default=0.015)
parser.add_argument('--model',default='lattice',help='lattice|lstm')
parser.add_argument('--skip_before_head',default=False)#in paper it's false
parser.add_argument('--hidden',default=100)
parser.add_argument('--momentum',default=0)
parser.add_argument('--bi',default=True)
parser.add_argument('--dataset',default='ontonote',help='resume|ontonote|weibo|msra')
parser.add_argument('--use_bigram',default=False)

parser.add_argument('--embed_dropout',default=0)
parser.add_argument('--output_dropout',default=0)
parser.add_argument('--epoch',default=100)
parser.add_argument('--seed',default=100)

args = parser.parse_args()

set_seed(args.seed)

fit_msg_list = [args.model,'bi' if args.bi else 'uni',str(args.batch)]
if args.model == 'lattice':
    fit_msg_list.append(str(args.skip_before_head))
fit_msg = ' '.join(fit_msg_list)
# fitlog.commit(__file__,fit_msg=fit_msg)


# fitlog.add_hyper(args)
device = torch.device(args.device)
for k,v in args.__dict__.items():
    print(k,v)

refresh_data = False
if args.dataset == 'ontonote':
    datasets,vocabs,embeddings = load_ontonotes4ner(ontonote4ner_cn_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    _refresh=refresh_data,index_token=False)
elif args.dataset == 'resume':
    datasets,vocabs,embeddings = load_resume_ner(resume_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    _refresh=refresh_data,index_token=False)
# exit()
w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                              _refresh=refresh_data)



cache_name = os.path.join('cache',args.dataset+'_lattice')
datasets,vocabs,embeddings = equip_chinese_ner_with_skip(datasets,vocabs,embeddings,w_list,yangjie_rich_pretrain_word_path,
                                                         _refresh=refresh_data,_cache_fp=cache_name)

print('中:embedding:{}'.format(embeddings['char'](24)))
print('embed lookup dropout:{}'.format(embeddings['word'].word_dropout))

# for k, v in datasets.items():
# #     v.apply_field(lambda x: list(map(len, x)), 'skips_l2r_word', 'lexicon_count')
# #     v.apply_field(lambda x:
# #                   list(map(lambda y:
# #                            list(map(lambda z: vocabs['word'].to_index(z), y)), x)),
# #                   'skips_l2r_word')

print(datasets['train'][0])
print('vocab info:')
for k,v in vocabs.items():
    print('{}:{}'.format(k,len(v)))
# print(datasets['dev'][0])
# print(datasets['test'][0])
# print(datasets['train'].get_all_fields().keys())
for k,v in datasets.items():
    if args.model == 'lattice':
        v.set_ignore_type('skips_l2r_word','skips_l2r_source','skips_r2l_word', 'skips_r2l_source')
        if args.skip_before_head:
            v.set_padder('skips_l2r_word',LatticeLexiconPadder())
            v.set_padder('skips_l2r_source',LatticeLexiconPadder())
            v.set_padder('skips_r2l_word',LatticeLexiconPadder())
            v.set_padder('skips_r2l_source',LatticeLexiconPadder(pad_val_dynamic=True))
        else:
            v.set_padder('skips_l2r_word',LatticeLexiconPadder())
            v.set_padder('skips_r2l_word', LatticeLexiconPadder())
            v.set_padder('skips_l2r_source', LatticeLexiconPadder(-1))
            v.set_padder('skips_r2l_source', LatticeLexiconPadder(pad_val_dynamic=True,dynamic_offset=1))
        if args.bi:
            v.set_input('chars','bigrams','seq_len',
                        'skips_l2r_word','skips_l2r_source','lexicon_count',
                        'skips_r2l_word', 'skips_r2l_source','lexicon_count_back',
                        'target',
                        use_1st_ins_infer_dim_type=True)
        else:
            v.set_input('chars','bigrams','seq_len',
                        'skips_l2r_word','skips_l2r_source','lexicon_count',
                        'target',
                        use_1st_ins_infer_dim_type=True)
        v.set_target('target','seq_len')

        v['target'].set_pad_val(0)
    elif args.model == 'lstm':
        v.set_ignore_type('skips_l2r_word','skips_l2r_source')
        v.set_padder('skips_l2r_word',LatticeLexiconPadder())
        v.set_padder('skips_l2r_source',LatticeLexiconPadder())
        v.set_input('chars','bigrams','seq_len','target',
                    use_1st_ins_infer_dim_type=True)
        v.set_target('target','seq_len')

        v['target'].set_pad_val(0)

print(datasets['dev']['skips_l2r_word'][100])


if args.model =='lattice':
    model = LatticeLSTM_SeqLabel_V1(embeddings['char'],embeddings['bigram'],embeddings['word'],
                        hidden_size=args.hidden,label_size=len(vocabs['label']),device=args.device,
                                 embed_dropout=args.embed_dropout,output_dropout=args.output_dropout,
                                 skip_batch_first=True,bidirectional=args.bi,debug=args.debug,
                                 skip_before_head=args.skip_before_head,use_bigram=args.use_bigram,
                                 vocabs=vocabs
                                 )
elif args.model == 'lstm':
    model = LSTM_SeqLabel(embeddings['char'],embeddings['bigram'],embeddings['word'],
                        hidden_size=args.hidden,label_size=len(vocabs['label']),device=args.device,
                          bidirectional=args.bi,
                          embed_dropout=args.embed_dropout,output_dropout=args.output_dropout,
                          use_bigram=args.use_bigram)

for k,v in model.state_dict().items():
    print('{}:{}'.format(k,v.size()))



# exit()
weight_dict = torch.load(open('/remote-home/xnli/weight_debug/lattice_yangjie.pkl','rb'))
# print(weight_dict.keys())
# for k,v in weight_dict.items():
#     print('{}:{}'.format(k,v.size()))
def state_dict_param(model):
    param_list = list(model.named_parameters())
    print(len(param_list))
    param_dict = {}
    for i in range(len(param_list)):
        param_dict[param_list[i][0]] = param_list[i][1]

    return param_dict


def copy_yangjie_lattice_weight(target,source_dict):
    t = state_dict_param(target)
    with torch.no_grad():
        t['encoder.char_cell.weight_ih'].set_(source_dict['lstm.forward_lstm.rnn.weight_ih'])
        t['encoder.char_cell.weight_hh'].set_(source_dict['lstm.forward_lstm.rnn.weight_hh'])
        t['encoder.char_cell.alpha_weight_ih'].set_(source_dict['lstm.forward_lstm.rnn.alpha_weight_ih'])
        t['encoder.char_cell.alpha_weight_hh'].set_(source_dict['lstm.forward_lstm.rnn.alpha_weight_hh'])
        t['encoder.char_cell.bias'].set_(source_dict['lstm.forward_lstm.rnn.bias'])
        t['encoder.char_cell.alpha_bias'].set_(source_dict['lstm.forward_lstm.rnn.alpha_bias'])
        t['encoder.word_cell.weight_ih'].set_(source_dict['lstm.forward_lstm.word_rnn.weight_ih'])
        t['encoder.word_cell.weight_hh'].set_(source_dict['lstm.forward_lstm.word_rnn.weight_hh'])
        t['encoder.word_cell.bias'].set_(source_dict['lstm.forward_lstm.word_rnn.bias'])

        t['encoder_back.char_cell.weight_ih'].set_(source_dict['lstm.backward_lstm.rnn.weight_ih'])
        t['encoder_back.char_cell.weight_hh'].set_(source_dict['lstm.backward_lstm.rnn.weight_hh'])
        t['encoder_back.char_cell.alpha_weight_ih'].set_(source_dict['lstm.backward_lstm.rnn.alpha_weight_ih'])
        t['encoder_back.char_cell.alpha_weight_hh'].set_(source_dict['lstm.backward_lstm.rnn.alpha_weight_hh'])
        t['encoder_back.char_cell.bias'].set_(source_dict['lstm.backward_lstm.rnn.bias'])
        t['encoder_back.char_cell.alpha_bias'].set_(source_dict['lstm.backward_lstm.rnn.alpha_bias'])
        t['encoder_back.word_cell.weight_ih'].set_(source_dict['lstm.backward_lstm.word_rnn.weight_ih'])
        t['encoder_back.word_cell.weight_hh'].set_(source_dict['lstm.backward_lstm.word_rnn.weight_hh'])
        t['encoder_back.word_cell.bias'].set_(source_dict['lstm.backward_lstm.word_rnn.bias'])

    for k,v in t.items():
        print('{}:{}'.format(k,v))

copy_yangjie_lattice_weight(model,weight_dict)

# print(vocabs['label'].word2idx.keys())








loss = LossInForward()

f1_metric = SpanFPreRecMetric(vocabs['label'],pred='pred',target='target',seq_len='seq_len',encoding_type='bmeso')
f1_metric_yj = SpanFPreRecMetric_YJ(vocabs['label'],pred='pred',target='target',seq_len='seq_len',encoding_type='bmesoyj')
acc_metric = AccuracyMetric(pred='pred',target='target',seq_len='seq_len')
metrics = [f1_metric,f1_metric_yj,acc_metric]

if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
elif args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum)



# tester = Tester(datasets['dev'],model,metrics=metrics,batch_size=args.test_batch,device=device)
# test_result = tester.test()
# print(test_result)
callbacks = [
    LRScheduler(lr_scheduler=LambdaLR(optimizer, lambda ep: 1 / (1 + 0.05)**ep))
]
print(datasets['train'][:2])
print(vocabs['char'].to_index('：'))
# StaticEmbedding
# datasets['train'] = datasets['train'][1:]
from fastNLP import SequentialSampler
trainer = Trainer(datasets['train'],model,
                  optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  dev_data=datasets['dev'],
                  device=device,
                  batch_size=args.batch,
                  n_epochs=args.epoch,
                  dev_batch_size=args.test_batch,
                  callbacks=callbacks,
                  check_code_level=-1,
                  sampler=SequentialSampler())

trainer.train()