import sys
sys.path.append('../../..')

from fastNLP.embeddings import CNNCharEmbedding, StaticEmbedding, StackEmbedding

from reproduction.sequence_labelling.ner.model.lstm_cnn_crf import CNNBiLSTMCRF
from fastNLP import Trainer
from fastNLP import SpanFPreRecMetric
from fastNLP import BucketSampler
from fastNLP import Const
from torch.optim import SGD
from fastNLP import GradientClipCallback
from fastNLP.core.callback import EvaluateCallback, LRScheduler
from torch.optim.lr_scheduler import LambdaLR
from fastNLP import cache_results

from fastNLP.io.pipe.conll import Conll2003NERPipe
encoding_type = 'bioes'
@cache_results('caches/conll2003_new.pkl', _refresh=True)
def load_data():
    # 替换路径
    paths = {'test':"NER/corpus/CoNLL-2003/eng.testb",
             'train':"NER/corpus/CoNLL-2003/eng.train",
             'dev':"NER/corpus/CoNLL-2003/eng.testa"}
    data = Conll2003NERPipe(encoding_type=encoding_type).process_from_file(paths)
    return data
data = load_data()
print(data)

char_embed = CNNCharEmbedding(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, filter_nums=[30],
                              kernel_sizes=[3], word_dropout=0, dropout=0.5)
word_embed = StaticEmbedding(vocab=data.get_vocab('words'),
                             model_dir_or_name='en-glove-6b-100d',
                             requires_grad=True, lower=True, word_dropout=0.01, dropout=0.5)
word_embed.embedding.weight.data = word_embed.embedding.weight.data/word_embed.embedding.weight.data.std()
embed = StackEmbedding([word_embed, char_embed])

model = CNNBiLSTMCRF(embed, hidden_size=200, num_layers=1, tag_vocab=data.vocabs[Const.TARGET],
                     encoding_type=encoding_type)

callbacks = [
            GradientClipCallback(clip_type='value', clip_value=5),
            EvaluateCallback(data=data.get_dataset('test'))  # 额外对test上的数据进行性能评测
            ]

optimizer = SGD(model.parameters(), lr=0.008, momentum=0.9)
scheduler = LRScheduler(LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch)))
callbacks.append(scheduler)

trainer = Trainer(train_data=data.get_dataset('train'), model=model, optimizer=optimizer, sampler=BucketSampler(),
                  device=0, dev_data=data.get_dataset('dev'), batch_size=20,
                  metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type),
                  callbacks=callbacks, num_workers=2, n_epochs=100, dev_batch_size=512)
trainer.train()