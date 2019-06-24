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
from fastNLP.core.callback import FitlogCallback, LRScheduler
from reproduction.seqence_labelling.ner.model.swats import SWATS

import fitlog
fitlog.debug()

from reproduction.seqence_labelling.ner.data.OntoNoteLoader import OntoNoteNERDataLoader

encoding_type = 'bioes'

data = OntoNoteNERDataLoader(encoding_type=encoding_type).process('/hdd/fudanNLP/fastNLP/others/data/v4/english',
                                                                  lower=True)

import joblib
raw_data = joblib.load('/hdd/fudanNLP/fastNLP/others/NER-with-LS/data/ontonotes_with_data.joblib')
def convert_to_ids(raw_words):
    ids = []
    for word in raw_words:
        id = raw_data['word_to_id'][word]
        id = raw_data['id_to_emb_map'][id]
        ids.append(id)
    return ids
word_embed = raw_data['emb_matrix']
for name, dataset in data.datasets.items():
    dataset.apply_field(convert_to_ids, field_name='raw_words', new_field_name=Const.INPUT)

print(data)
char_embed = CNNCharEmbedding(vocab=data.vocabs['cap_words'], embed_size=30, char_emb_size=30, filter_nums=[30],
                              kernel_sizes=[3])
# word_embed = StaticEmbedding(vocab=data.vocabs[Const.INPUT],
#                              model_dir_or_name='/hdd/fudanNLP/pretrain_vectors/glove.6B.100d.txt',
#                              requires_grad=True)

model = CNNBiLSTMCRF(word_embed, char_embed, hidden_size=1200, num_layers=1, tag_vocab=data.vocabs[Const.TARGET],
                     encoding_type=encoding_type)

callbacks = [GradientClipCallback(clip_value=5, clip_type='value'),
             FitlogCallback(data.datasets['test'], verbose=1)]

optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = LRScheduler(LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch)))
callbacks.append(scheduler)
# optimizer = SWATS(model.parameters(), verbose=True)
# optimizer = Adam(model.parameters(), lr=0.005)


trainer = Trainer(train_data=data.datasets['train'], model=model, optimizer=optimizer, sampler=BucketSampler(num_buckets=100),
                  device=0, dev_data=data.datasets['dev'], batch_size=10,
                  metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type),
                  callbacks=callbacks, num_workers=1, n_epochs=100)
trainer.train()