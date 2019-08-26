import sys
sys.path.append('../..')

from reproduction.joint_cws_parse.data.data_loader import CTBxJointLoader
from fastNLP.embeddings.static_embedding import StaticEmbedding
from torch import nn
from functools import partial
from reproduction.joint_cws_parse.models.CharParser import CharParser
from reproduction.joint_cws_parse.models.metrics import SegAppCharParseF1Metric, CWSMetric
from fastNLP import BucketSampler, Trainer
from torch import optim
from reproduction.joint_cws_parse.models.callbacks import DevCallback
from torch.optim.lr_scheduler import StepLR
from fastNLP import Tester
from fastNLP import GradientClipCallback, LRScheduler
import os
from fastNLP import cache_results

def set_random_seed(random_seed=666):
    import random, numpy, torch
    random.seed(random_seed)
    numpy.random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)

uniform_init = partial(nn.init.normal_, std=0.02)

###################################################
# 需要变动的超参放到这里
lr = 0.002   # 0.01~0.001
dropout = 0.33  # 0.3~0.6
weight_decay = 0  # 1e-5, 1e-6, 0
arc_mlp_size = 500   # 200, 300
rnn_hidden_size = 400  # 200, 300, 400
rnn_layers = 3  # 2, 3
encoder = 'var-lstm'  # var-lstm, lstm
emb_size = 100  # 64 , 100
label_mlp_size = 100

batch_size = 32
update_every = 4
n_epochs = 100
data_name = 'new_ctb7'
####################################################
data_folder = f'/remote-home/hyan01/exps/JointCwsPosParser/data/{data_name}/output'  # 填写在数据所在文件夹, 文件夹下应该有train, dev, test等三个文件
vector_folder = '/remote-home/hyan01/exps/CWS/pretrain/vectors'  # 预训练的vector，下面应该包含三个文件: 1grams_t3_m50_corpus.txt, 2grams_t3_m50_corpus.txt, 3grams_t3_m50_corpus.txt

set_random_seed(1234)
device = 0

@cache_results('caches/{}.pkl'.format(data_name))
def get_data():
    data = CTBxJointLoader().process(data_folder)
    char_labels_vocab = data.vocabs['char_labels']

    pre_chars_vocab = data.vocabs['pre_chars']
    pre_bigrams_vocab = data.vocabs['pre_bigrams']
    pre_trigrams_vocab = data.vocabs['pre_trigrams']

    chars_vocab = data.vocabs['chars']
    bigrams_vocab = data.vocabs['bigrams']
    trigrams_vocab = data.vocabs['trigrams']
    pre_chars_embed = StaticEmbedding(pre_chars_vocab,
                                      model_dir_or_name=os.path.join(vector_folder, '1grams_t3_m50_corpus.txt'),
                                      init_method=uniform_init, normalize=False)
    pre_chars_embed.embedding.weight.data = pre_chars_embed.embedding.weight.data / pre_chars_embed.embedding.weight.data.std()
    pre_bigrams_embed = StaticEmbedding(pre_bigrams_vocab,
                                        model_dir_or_name=os.path.join(vector_folder, '2grams_t3_m50_corpus.txt'),
                                        init_method=uniform_init, normalize=False)
    pre_bigrams_embed.embedding.weight.data = pre_bigrams_embed.embedding.weight.data / pre_bigrams_embed.embedding.weight.data.std()
    pre_trigrams_embed = StaticEmbedding(pre_trigrams_vocab,
                                         model_dir_or_name=os.path.join(vector_folder, '3grams_t3_m50_corpus.txt'),
                                         init_method=uniform_init, normalize=False)
    pre_trigrams_embed.embedding.weight.data = pre_trigrams_embed.embedding.weight.data / pre_trigrams_embed.embedding.weight.data.std()

    return chars_vocab, bigrams_vocab, trigrams_vocab, char_labels_vocab, pre_chars_embed, pre_bigrams_embed, pre_trigrams_embed, data

chars_vocab, bigrams_vocab, trigrams_vocab, char_labels_vocab, pre_chars_embed, pre_bigrams_embed, pre_trigrams_embed, data = get_data()

print(data)
model = CharParser(char_vocab_size=len(chars_vocab),
                    emb_dim=emb_size,
                    bigram_vocab_size=len(bigrams_vocab),
                   trigram_vocab_size=len(trigrams_vocab),
                    num_label=len(char_labels_vocab),
                    rnn_layers=rnn_layers,
                    rnn_hidden_size=rnn_hidden_size,
                    arc_mlp_size=arc_mlp_size,
                    label_mlp_size=label_mlp_size,
                    dropout=dropout,
                    encoder=encoder,
                    use_greedy_infer=False,
                    app_index=char_labels_vocab['APP'],
                     pre_chars_embed=pre_chars_embed,
                   pre_bigrams_embed=pre_bigrams_embed,
                   pre_trigrams_embed=pre_trigrams_embed)

metric1 = SegAppCharParseF1Metric(char_labels_vocab['APP'])
metric2 = CWSMetric(char_labels_vocab['APP'])
metrics = [metric1, metric2]

optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr,
                       weight_decay=weight_decay, betas=[0.9, 0.9])

sampler = BucketSampler(seq_len_field_name='seq_lens')
callbacks = []

from fastNLP.core.callback import Callback
from torch.optim.lr_scheduler import LambdaLR
class SchedulerCallback(Callback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def on_backward_end(self):
        if self.step % self.update_every==0:
            self.scheduler.step()

scheduler = LambdaLR(optimizer, lr_lambda=lambda step:(0.75)**(step//5000))
# scheduler = LambdaLR(optimizer, lr_lambda=lambda step:(0.75)**(step//5000))
# scheduler = StepLR(optimizer, step_size=18, gamma=0.75)
scheduler_callback = SchedulerCallback(scheduler)
# callbacks.append(optim_callback)
# scheduler_callback = LRScheduler(scheduler)
callbacks.append(scheduler_callback)
callbacks.append(GradientClipCallback(clip_type='value', clip_value=5))

tester = Tester(data=data.datasets['test'], model=model, metrics=metrics,
                    batch_size=64, device=device, verbose=0)
dev_callback = DevCallback(tester)
callbacks.append(dev_callback)

trainer = Trainer(data.datasets['train'], model, loss=None, metrics=metrics, n_epochs=n_epochs, batch_size=batch_size, print_every=3,
                 validate_every=-1, dev_data=data.datasets['dev'], save_path=None, optimizer=optimizer,
                 check_code_level=0, metric_key='u_f1', sampler=sampler, num_workers=2, use_tqdm=True,
                 device=device, callbacks=callbacks, update_every=update_every)
trainer.train()