

from fastNLP.modules.encoder.embedding import CNNCharEmbedding, StaticEmbedding, BertEmbedding
from fastNLP.core.vocabulary import VocabularyOption

from reproduction.seqence_labelling.ner.model.lstm_cnn_crf import CNNBiLSTMCRF
from fastNLP import Trainer
from fastNLP import SpanFPreRecMetric
from fastNLP import BucketSampler
from fastNLP import Const
from torch.optim import SGD, Adam
from fastNLP import GradientClipCallback
from fastNLP.core.callback import FitlogCallback, LRScheduler
from torch.optim.lr_scheduler import LambdaLR
import fitlog
fitlog.debug()

from reproduction.seqence_labelling.ner.data.Conll2003Loader import Conll2003DataLoader

encoding_type = 'bioes'

data = Conll2003DataLoader(encoding_type=encoding_type).process('/hdd/fudanNLP/fastNLP/others/data/conll2003',
                                                                word_vocab_opt=VocabularyOption(min_freq=2))
print(data)
char_embed = CNNCharEmbedding(vocab=data.vocabs['cap_words'], embed_size=30, char_emb_size=30, filter_nums=[30],
                              kernel_sizes=[3])
word_embed = StaticEmbedding(vocab=data.vocabs[Const.INPUT],
                             model_dir_or_name='/hdd/fudanNLP/pretrain_vectors/glove.6B.100d.txt',
                             requires_grad=True)
word_embed.embedding.weight.data = word_embed.embedding.weight.data/word_embed.embedding.weight.data.std()

model = CNNBiLSTMCRF(word_embed, char_embed, hidden_size=200, num_layers=1, tag_vocab=data.vocabs[Const.TARGET],
                     encoding_type=encoding_type)

optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = LRScheduler(LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch)))

callbacks = [GradientClipCallback(clip_type='value', clip_value=5), FitlogCallback({'test':data.datasets['test'],
                                                                      'train':data.datasets['train']}, verbose=1),
             scheduler]

trainer = Trainer(train_data=data.datasets['train'], model=model, optimizer=optimizer, sampler=BucketSampler(),
                  device=0, dev_data=data.datasets['dev'], batch_size=10,
                  metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type),
                  callbacks=callbacks, num_workers=1, n_epochs=100)
trainer.train()