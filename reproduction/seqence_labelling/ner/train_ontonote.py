

from fastNLP.modules.encoder.embedding import CNNCharEmbedding, StaticEmbedding

from reproduction.seqence_labelling.ner.model.lstm_cnn_crf import CNNBiLSTMCRF
from fastNLP import Trainer
from fastNLP import SpanFPreRecMetric
from fastNLP import BucketSampler
from fastNLP import Const
from torch.optim import SGD, Adam
from fastNLP import GradientClipCallback
from fastNLP.core.callback import FitlogCallback
import fitlog
fitlog.debug()

from reproduction.seqence_labelling.ner.data.OntoNoteLoader import OntoNoteNERDataLoader

encoding_type = 'bioes'

data = OntoNoteNERDataLoader(encoding_type=encoding_type).process('/hdd/fudanNLP/fastNLP/others/data/v4/english')
print(data)
char_embed = CNNCharEmbedding(vocab=data.vocabs['cap_words'], embed_size=30, char_emb_size=30, filter_nums=[30],
                              kernel_sizes=[3])
word_embed = StaticEmbedding(vocab=data.vocabs[Const.INPUT],
                             model_dir_or_name='/hdd/fudanNLP/pretrain_vectors/glove.6B.100d.txt',
                             requires_grad=True)

model = CNNBiLSTMCRF(word_embed, char_embed, hidden_size=400, num_layers=2, tag_vocab=data.vocabs[Const.TARGET],
                     encoding_type=encoding_type)

optimizer = SGD(model.parameters(), lr=0.015, momentum=0.9)

callbacks = [GradientClipCallback(), FitlogCallback(data.datasets['test'], verbose=1)]

trainer = Trainer(train_data=data.datasets['train'], model=model, optimizer=optimizer, sampler=BucketSampler(),
                  device=1, dev_data=data.datasets['dev'], batch_size=32,
                  metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type),
                  callbacks=callbacks, num_workers=1, n_epochs=100)
trainer.train()