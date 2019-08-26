from reproduction.Star_transformer.util import get_argparser, set_gpu, set_rng_seeds, add_model_args
seed = set_rng_seeds(15360)
print('RNG SEED {}'.format(seed))
from reproduction.Star_transformer.datasets import load_seqtag, load_sst, load_snli, EmbedLoader, MAX_LEN
import torch.nn as nn
import torch
import numpy as np
import fastNLP as FN
from fastNLP.models.star_transformer import STSeqLabel, STSeqCls, STNLICls
from fastNLP.core.const import Const as C
import sys
#sys.path.append('/remote-home/yfshao/workdir/dev_fastnlp/')
import os
pre_dir = os.path.join(os.environ['HOME'], 'workdir/datasets/')

g_model_select = {
    'pos': STSeqLabel,
    'ner': STSeqLabel,
    'cls': STSeqCls,
    'nli': STNLICls,
}

g_emb_file_path = {'en': pre_dir + 'word_vector/glove.840B.300d.txt',
                   'zh': pre_dir + 'cc.zh.300.vec'}

g_args = None
g_model_cfg = None


def get_ptb_pos():
    pos_dir = '/remote-home/yfshao/workdir/datasets/pos'
    pos_files = ['train.pos', 'dev.pos', 'test.pos', ]
    return load_seqtag(pos_dir, pos_files, [0, 1])


def get_ctb_pos():
    ctb_dir = '/remote-home/yfshao/workdir/datasets/ctb9_hy'
    files = ['train.conllx', 'dev.conllx', 'test.conllx']
    return load_seqtag(ctb_dir, files, [1, 4])


def get_conll2012_pos():
    path = '/remote-home/yfshao/workdir/datasets/ontonotes/pos'
    files = ['ontonotes-conll.train',
             'ontonotes-conll.dev',
             'ontonotes-conll.conll-2012-test']
    return load_seqtag(path, files, [0, 1])


def get_conll2012_ner():
    path = '/remote-home/yfshao/workdir/datasets/ontonotes/ner'
    files = ['bieso-ontonotes-conll-ner.train',
             'bieso-ontonotes-conll-ner.dev',
             'bieso-ontonotes-conll-ner.conll-2012-test']
    return load_seqtag(path, files, [0, 1])


def get_sst():
    path = pre_dir + 'SST'
    files = ['train.txt', 'dev.txt', 'test.txt']
    return load_sst(path, files)


def get_snli():
    path = '/remote-home/yfshao/workdir/datasets/nli-data/snli_1.0'
    files = ['snli_1.0_train.jsonl',
             'snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl']
    return load_snli(path, files)


g_datasets = {
    'ptb-pos': get_ptb_pos,
    'ctb-pos': get_ctb_pos,
    'conll-pos': get_conll2012_pos,
    'conll-ner': get_conll2012_ner,
    'sst-cls': get_sst,
    'snli-nli': get_snli,
}


def load_pretrain_emb(word_v, lang='en'):
    print('loading pre-train embeddings')
    emb = EmbedLoader.fast_load_embedding(300, g_emb_file_path[lang], word_v)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    emb = torch.tensor(emb, dtype=torch.float32)
    print('embedding mean: {:.6}, std: {:.6}'.format(emb.mean(), emb.std()))
    emb[word_v.padding_idx].fill_(0)
    return emb


class MyCallback(FN.core.callback.Callback):
    def on_train_begin(self):
        super(MyCallback, self).on_train_begin()
        self.init_lrs = [pg['lr'] for pg in self.optimizer.param_groups]

    def on_backward_end(self):
        nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 5.0)

    def on_step_end(self):
        return 
        warm_steps = 6000
        # learning rate warm-up & decay
        if self.step <= warm_steps:
            for lr, pg in zip(self.init_lrs, self.optimizer.param_groups):
                pg['lr'] = lr * (self.step / float(warm_steps))

        elif self.step % 3000 == 0:
            for pg in self.optimizer.param_groups:
                cur_lr = pg['lr']
                pg['lr'] = max(1e-5, cur_lr*g_args.lr_decay)



def train():
    print('loading data')
    ds_list, word_v, tag_v = g_datasets['{}-{}'.format(
        g_args.ds, g_args.task)]()
    print(ds_list[0][:2])
    print(len(ds_list[0]), len(ds_list[1]), len(ds_list[2]))
    embed = load_pretrain_emb(word_v, lang='zh' if g_args.ds == 'ctb' else 'en')
    g_model_cfg['num_cls'] = len(tag_v)
    print(g_model_cfg)
    g_model_cfg['init_embed'] = embed
    model = g_model_select[g_args.task.lower()](**g_model_cfg)

    def init_model(model):
        for p in model.parameters():
            if p.size(0) != len(word_v):
                if len(p.size())<2:
                    nn.init.constant_(p, 0.0)
                else:
                    nn.init.normal_(p, 0.0, 0.05)
    init_model(model)
    train_data = ds_list[0]
    dev_data = ds_list[1]
    test_data = ds_list[2]
    print(tag_v.word2idx)

    if g_args.task in ['pos', 'ner']:
        padding_idx = tag_v.padding_idx
    else:
        padding_idx = -100
    print('padding_idx ', padding_idx)
    loss = FN.CrossEntropyLoss(padding_idx=padding_idx)
    metrics = {
        'pos': (None, FN.AccuracyMetric()),
        'ner': ('f', FN.core.metrics.SpanFPreRecMetric(
            tag_vocab=tag_v, encoding_type='bmeso', ignore_labels=[''], )),
        'cls': (None, FN.AccuracyMetric()),
        'nli': (None, FN.AccuracyMetric()),
    }
    metric_key, metric = metrics[g_args.task]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    params = [(x,y) for x,y in list(model.named_parameters()) if y.requires_grad and y.size(0) != len(word_v)]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    print([n for n,p in params])
    optim_cfg = [
        #{'params': model.enc.embedding.parameters(), 'lr': g_args.lr*0.1},
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'lr': g_args.lr, 'weight_decay': 1.0*g_args.w_decay},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'lr': g_args.lr, 'weight_decay': 0.0*g_args.w_decay}
    ]

    print(model)
    trainer = FN.Trainer(model=model, train_data=train_data, dev_data=dev_data,
                         loss=loss, metrics=metric, metric_key=metric_key,
                         optimizer=torch.optim.Adam(optim_cfg),
                         n_epochs=g_args.ep, batch_size=g_args.bsz, print_every=100, validate_every=1000,
                         device=device,
                         use_tqdm=False, prefetch=False,
                         save_path=g_args.log,
                         sampler=FN.BucketSampler(100, g_args.bsz, C.INPUT_LEN),
                         callbacks=[MyCallback()])

    print(trainer.train())
    tester = FN.Tester(data=test_data, model=model, metrics=metric,
                       batch_size=128, device=device)
    print(tester.test())


def test():
    pass


def infer():
    pass


run_select = {
    'train': train,
    'test': test,
    'infer': infer,
}


def main():
    global g_args, g_model_cfg
    import signal

    def signal_handler(signal, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    parser = get_argparser()
    parser.add_argument('--task', choices=['pos', 'ner', 'cls', 'nli'])
    parser.add_argument('--mode', choices=['train', 'test', 'infer'])
    parser.add_argument('--ds', type=str)
    add_model_args(parser)
    g_args = parser.parse_args()
    print(g_args.__dict__)
    set_gpu(g_args.gpu)
    g_model_cfg = {
        'init_embed': (None, 300),
        'num_cls': None,
        'hidden_size': g_args.hidden,
        'num_layers': 2,
        'num_head': g_args.nhead,
        'head_dim': g_args.hdim,
        'max_len': MAX_LEN,
        'cls_hidden_size': 200,
        'emb_dropout': g_args.drop,
        'dropout': g_args.drop,
    }
    run_select[g_args.mode.lower()]()


if __name__ == '__main__':
    main()
