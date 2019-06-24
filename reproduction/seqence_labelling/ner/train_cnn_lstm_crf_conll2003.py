

from fastNLP.modules.encoder.embedding import CNNCharEmbedding, StaticEmbedding, BertEmbedding, ElmoEmbedding, LSTMCharEmbedding
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
from reproduction.seqence_labelling.ner.model.swats import SWATS

import fitlog
fitlog.debug()

from reproduction.seqence_labelling.ner.data.Conll2003Loader import Conll2003DataLoader

encoding_type = 'bioes'

data = Conll2003DataLoader(encoding_type=encoding_type).process('../../../../others/data/conll2003',
                                                                word_vocab_opt=VocabularyOption(min_freq=2),
                                                                lower=False)
print(data)
char_embed = CNNCharEmbedding(vocab=data.vocabs['cap_words'], embed_size=30, char_emb_size=30, filter_nums=[30],
                              kernel_sizes=[3])
# char_embed = LSTMCharEmbedding(vocab=data.vocabs['cap_words'], embed_size=30 ,char_emb_size=30)
word_embed = StaticEmbedding(vocab=data.vocabs[Const.INPUT],
                             model_dir_or_name='/hdd/fudanNLP/pretrain_vectors/wiki_en_100_50_case_2.txt',
                             requires_grad=True)
word_embed.embedding.weight.data = word_embed.embedding.weight.data/word_embed.embedding.weight.data.std()

# import joblib
# raw_data = joblib.load('/hdd/fudanNLP/fastNLP/others/NER-with-LS/data/conll_with_data.joblib')
# def convert_to_ids(raw_words):
#     ids = []
#     for word in raw_words:
#         id = raw_data['word_to_id'][word]
#         id = raw_data['id_to_emb_map'][id]
#         ids.append(id)
#     return ids
# word_embed = raw_data['emb_matrix']
# for name, dataset in data.datasets.items():
#     dataset.apply_field(convert_to_ids, field_name='raw_words', new_field_name=Const.INPUT)

# word_embed = ElmoEmbedding(vocab=data.vocabs['cap_words'],
#                              model_dir_or_name='/hdd/fudanNLP/fastNLP/others/pretrained_models/elmo_en',
#                              requires_grad=True)

model = CNNBiLSTMCRF(word_embed, char_embed, hidden_size=200, num_layers=1, tag_vocab=data.vocabs[Const.TARGET],
                     encoding_type=encoding_type)

callbacks = [
            GradientClipCallback(clip_type='value', clip_value=5)
            , FitlogCallback({'test':data.datasets['test']}, verbose=1)
            ]
# optimizer = Adam(model.parameters(), lr=0.005)
optimizer = SWATS(model.parameters(), verbose=True)
# optimizer = SGD(model.parameters(), lr=0.008, momentum=0.9)
# scheduler = LRScheduler(LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch)))
# callbacks.append(scheduler)

trainer = Trainer(train_data=data.datasets['train'], model=model, optimizer=optimizer, sampler=BucketSampler(),
                  device=1, dev_data=data.datasets['dev'], batch_size=10,
                  metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type),
                  callbacks=callbacks, num_workers=1, n_epochs=100)
trainer.train()