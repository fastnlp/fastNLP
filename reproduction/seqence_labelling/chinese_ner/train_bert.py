

"""
使用Bert进行中文命名实体识别

"""

import sys

sys.path.append('../../../')

from torch import nn

from fastNLP.embeddings import BertEmbedding, Embedding
from reproduction.seqence_labelling.chinese_ner.data.ChineseNER import ChineseNERLoader
from fastNLP import Trainer, Const
from fastNLP import BucketSampler, SpanFPreRecMetric, GradientClipCallback
from fastNLP.modules import MLP
from fastNLP.core.callback import WarmupCallback
from fastNLP import CrossEntropyLoss
from fastNLP.core.optimizer import AdamW
import os

from fastNLP import cache_results

encoding_type = 'bio'

@cache_results('caches/msra.pkl')
def get_data():
    data = ChineseNERLoader(encoding_type=encoding_type).process("MSRA/")
    return data
data = get_data()
print(data)

class BertCNNER(nn.Module):
    def __init__(self, embed, tag_size):
        super().__init__()

        self.embedding = Embedding(embed, dropout=0.1)
        self.tag_size = tag_size
        self.mlp = MLP(size_layer=[self.embedding.embedding_dim, tag_size])
    def forward(self, chars):
        # batch_size, max_len = words.size()
        chars = self.embedding(chars)
        outputs = self.mlp(chars)

        return {Const.OUTPUT: outputs}

embed = BertEmbedding(data.vocabs[Const.CHAR_INPUT], model_dir_or_name='en-base',
                        pool_method='max', requires_grad=True, layers='11')

for name, dataset in data.datasets.items():
    dataset.set_pad_val(Const.TARGET, -100)

callbacks = [
                GradientClipCallback(clip_type='norm', clip_value=1),
                WarmupCallback(warmup=0.1, schedule='linear')
            ]

model = BertCNNER(embed, len(data.vocabs[Const.TARGET]))
optimizer = AdamW(model.parameters(), lr=1e-4)

for name, dataset in data.datasets.items():
    original_len = len(dataset)
    dataset.drop(lambda x:x['seq_len']>256, inplace=True)
    clipped_len = len(dataset)
    print("Delete {} instances in {}.".format(original_len-clipped_len, name))

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

trainer = Trainer(train_data=data.datasets['train'], model=model, optimizer=optimizer, sampler=BucketSampler(),
                  device=[0, 1], dev_data=data.datasets['test'], batch_size=20,
                  metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type),
                  loss=CrossEntropyLoss(reduction='sum'),
                  callbacks=callbacks, num_workers=2, n_epochs=5,
                  check_code_level=-1, update_every=3)
trainer.train()

