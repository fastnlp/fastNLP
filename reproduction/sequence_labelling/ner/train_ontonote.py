import sys

sys.path.append('../../..')

from fastNLP.embeddings import CNNCharEmbedding, StaticEmbedding, StackEmbedding

from reproduction.sequence_labelling.ner.model.lstm_cnn_crf import CNNBiLSTMCRF
from fastNLP import Trainer
from fastNLP import SpanFPreRecMetric
from fastNLP import Const
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from fastNLP import GradientClipCallback
from fastNLP import BucketSampler
from fastNLP.core.callback import EvaluateCallback, LRScheduler
from fastNLP import cache_results
from fastNLP.io.pipe.conll import OntoNotesNERPipe

#######hyper
normalize = False
lr = 0.01
dropout = 0.5
batch_size = 32
data_name = 'ontonote'
#######hyper


encoding_type = 'bioes'

@cache_results('caches/ontonotes.pkl', _refresh=True)
def cache():
    data = OntoNotesNERPipe(encoding_type=encoding_type).process_from_file('../../../../others/data/v4/english')
    char_embed = CNNCharEmbedding(vocab=data.vocabs['words'], embed_size=30, char_emb_size=30, filter_nums=[30],
                                  kernel_sizes=[3], dropout=dropout)
    word_embed = StaticEmbedding(vocab=data.vocabs[Const.INPUT],
                                 model_dir_or_name='en-glove-6b-100d',
                                 requires_grad=True,
                                 normalize=normalize,
                                 word_dropout=0.01,
                                 dropout=dropout,
                                 lower=True,
                                 min_freq=1)
    return data, char_embed, word_embed
data, char_embed, word_embed = cache()

print(data)

embed = StackEmbedding([word_embed, char_embed])
model = CNNBiLSTMCRF(embed, hidden_size=1200, num_layers=1, tag_vocab=data.vocabs[Const.TARGET],
                     encoding_type=encoding_type, dropout=dropout)

callbacks = [
                GradientClipCallback(clip_value=5, clip_type='value'),
                EvaluateCallback(data.datasets['test'])
             ]

optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = LRScheduler(LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch)))
callbacks.append(scheduler)


trainer = Trainer(train_data=data.get_dataset('train'), model=model, optimizer=optimizer, sampler=BucketSampler(num_buckets=100),
                  device=0, dev_data=data.get_dataset('dev'), batch_size=batch_size,
                  metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type),
                  callbacks=callbacks, num_workers=1, n_epochs=100, dev_batch_size=256)
trainer.train()