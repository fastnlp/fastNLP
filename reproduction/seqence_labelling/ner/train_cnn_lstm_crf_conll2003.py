import sys
sys.path.append('../../..')

from fastNLP.modules.encoder.embedding import CNNCharEmbedding, StaticEmbedding, BertEmbedding, ElmoEmbedding, StackEmbedding
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
from fastNLP.core.optimizer import AdamW
# from reproduction.seqence_labelling.ner.model.swats import SWATS
from reproduction.seqence_labelling.chinese_ner.callbacks import SaveModelCallback
from fastNLP import cache_results

import fitlog
fitlog.debug()

from reproduction.seqence_labelling.ner.data.Conll2003Loader import Conll2003DataLoader

encoding_type = 'bioes'
@cache_results('caches/upper_conll2003.pkl')
def load_data():
    data = Conll2003DataLoader(encoding_type=encoding_type).process('../../../../others/data/conll2003',
                                                                       word_vocab_opt=VocabularyOption(min_freq=1),
                                                                       lower=False)
    return data
data = load_data()
print(data)
char_embed = CNNCharEmbedding(vocab=data.vocabs['words'], embed_size=30, char_emb_size=30, filter_nums=[30],
                              kernel_sizes=[3], word_dropout=0.01, dropout=0.5)
# char_embed = LSTMCharEmbedding(vocab=data.vocabs['cap_words'], embed_size=30 ,char_emb_size=30)
word_embed = StaticEmbedding(vocab=data.vocabs['words'],
                             model_dir_or_name='/hdd/fudanNLP/pretrain_vectors/glove.6B.100d.txt',
                             requires_grad=True, lower=True, word_dropout=0.01, dropout=0.5)
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

# elmo_embed = ElmoEmbedding(vocab=data.vocabs['cap_words'],
#                              model_dir_or_name='.',
#                              requires_grad=True, layers='mix')
# char_embed = StackEmbedding([elmo_embed, char_embed])

model = CNNBiLSTMCRF(word_embed, char_embed, hidden_size=200, num_layers=1, tag_vocab=data.vocabs[Const.TARGET],
                     encoding_type=encoding_type)

callbacks = [
            GradientClipCallback(clip_type='value', clip_value=5),
            FitlogCallback({'test':data.datasets['test']}, verbose=1),
            # SaveModelCallback('save_models/', top=3, only_param=False, save_on_exception=True)
            ]
# optimizer = Adam(model.parameters(), lr=0.001)
# optimizer = SWATS(model.parameters(), verbose=True)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = LRScheduler(LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch)))
callbacks.append(scheduler)


trainer = Trainer(train_data=data.datasets['train'], model=model, optimizer=optimizer, sampler=BucketSampler(batch_size=20),
                  device=1, dev_data=data.datasets['dev'], batch_size=20,
                  metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type),
                  callbacks=callbacks, num_workers=2, n_epochs=100)
trainer.train()