

"""
使用Bert进行英文命名实体识别

"""

import sys

sys.path.append('../../../')

from reproduction.sequence_labelling.ner.model.bert_crf import BertCRF
from fastNLP.embeddings import BertEmbedding
from fastNLP import Trainer, Const
from fastNLP import BucketSampler, SpanFPreRecMetric, GradientClipCallback
from fastNLP.core.callback import WarmupCallback
from fastNLP.core.optimizer import AdamW
from fastNLP.io import Conll2003NERPipe

from fastNLP import cache_results, EvaluateCallback

encoding_type = 'bioes'

@cache_results('caches/conll2003.pkl', _refresh=False)
def load_data():
    # 替换路径
    paths = 'data/conll2003'
    data = Conll2003NERPipe(encoding_type=encoding_type).process_from_file(paths)
    return data
data = load_data()
print(data)

embed = BertEmbedding(data.get_vocab(Const.INPUT), model_dir_or_name='en-base-cased',
                        pool_method='max', requires_grad=True, layers='11', include_cls_sep=False, dropout=0.5,
                      word_dropout=0.01)

callbacks = [
                GradientClipCallback(clip_type='norm', clip_value=1),
                WarmupCallback(warmup=0.1, schedule='linear'),
                EvaluateCallback(data.get_dataset('test'))
            ]

model = BertCRF(embed, tag_vocab=data.get_vocab('target'), encoding_type=encoding_type)
optimizer = AdamW(model.parameters(), lr=2e-5)

trainer = Trainer(train_data=data.datasets['train'], model=model, optimizer=optimizer, sampler=BucketSampler(),
                  device=0, dev_data=data.datasets['dev'], batch_size=6,
                  metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type),
                  loss=None, callbacks=callbacks, num_workers=2, n_epochs=5,
                  check_code_level=0, update_every=3, test_use_tqdm=False)
trainer.train()

