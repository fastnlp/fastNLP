import sys

sys.path.append('../../..')

from fastNLP.modules.encoder.embedding import CNNCharEmbedding, StaticEmbedding

from reproduction.seqence_labelling.ner.model.lstm_cnn_crf import CNNBiLSTMCRF
from fastNLP import Trainer
from fastNLP import SpanFPreRecMetric
from fastNLP import BucketSampler
from fastNLP import Const
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from fastNLP import GradientClipCallback
from fastNLP.core.vocabulary import VocabularyOption
from fastNLP.core.callback import FitlogCallback, LRScheduler
from functools import partial
from torch import nn
from fastNLP import cache_results

import fitlog
fitlog.debug()
fitlog.set_log_dir('logs/')

fitlog.add_hyper_in_file(__file__)
#######hyper
normalize = False
divide_std = True
lower = False
lr = 0.015
dropout = 0.5
batch_size = 20
init_method = 'default'
job_embed = False
data_name = 'ontonote'
#######hyper


init_method = {'default': None,
               'xavier': partial(nn.init.xavier_normal_, gain=0.02),
               'normal': partial(nn.init.normal_, std=0.02)
               }[init_method]


from reproduction.seqence_labelling.ner.data.OntoNoteLoader import OntoNoteNERDataLoader

encoding_type = 'bioes'

@cache_results('caches/ontonotes.pkl')
def cache():
    data = OntoNoteNERDataLoader(encoding_type=encoding_type).process('../../../../others/data/v4/english',
                                                                      lower=lower,
                                                                      word_vocab_opt=VocabularyOption(min_freq=1))
    char_embed = CNNCharEmbedding(vocab=data.vocabs['cap_words'], embed_size=30, char_emb_size=30, filter_nums=[30],
                                  kernel_sizes=[3])
    word_embed = StaticEmbedding(vocab=data.vocabs[Const.INPUT],
                                 model_dir_or_name='/remote-home/hyan01/fastnlp_caches/glove.6B.100d/glove.6B.100d.txt',
                                 requires_grad=True,
                                 normalize=normalize,
                                 init_method=init_method)
    return data, char_embed, word_embed
data, char_embed, word_embed = cache()

print(data)

model = CNNBiLSTMCRF(word_embed, char_embed, hidden_size=1200, num_layers=1, tag_vocab=data.vocabs[Const.TARGET],
                     encoding_type=encoding_type, dropout=dropout)

callbacks = [
                GradientClipCallback(clip_value=5, clip_type='value'),
                FitlogCallback(data.datasets['test'], verbose=1)
             ]

optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = LRScheduler(LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch)))
callbacks.append(scheduler)


trainer = Trainer(train_data=data.datasets['dev'][:100], model=model, optimizer=optimizer, sampler=None,
                  device=0, dev_data=data.datasets['dev'][:100], batch_size=batch_size,
                  metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type),
                  callbacks=callbacks, num_workers=1, n_epochs=100)
trainer.train()