

"""
使用Bert进行中文命名实体识别

"""

import sys

sys.path.append('../../../')

from torch import nn

from fastNLP.embeddings import BertEmbedding, Embedding
from fastNLP import Trainer, Const
from fastNLP import BucketSampler, SpanFPreRecMetric, GradientClipCallback
from fastNLP.modules import MLP
from fastNLP.core.callback import WarmupCallback
from fastNLP import CrossEntropyLoss
from fastNLP.core.optimizer import AdamW
from fastNLP.io import MsraNERPipe, MsraNERLoader, WeiboNERPipe

from fastNLP import cache_results

encoding_type = 'bio'

@cache_results('caches/weibo.pkl', _refresh=False)
def get_data():
    # data_dir = MsraNERLoader().download(dev_ratio=0)
    # data = MsraNERPipe(encoding_type=encoding_type, target_pad_val=-100).process_from_file(data_dir)
    data = WeiboNERPipe(encoding_type=encoding_type).process_from_file()
    return data
data = get_data()
print(data)

class BertCNNER(nn.Module):
    def __init__(self, embed, tag_size):
        super().__init__()
        self.embedding = embed
        self.tag_size = tag_size
        self.mlp = MLP(size_layer=[self.embedding.embedding_dim, tag_size])

    def forward(self, chars):
        # batch_size, max_len = words.size()
        chars = self.embedding(chars)
        outputs = self.mlp(chars)

        return {Const.OUTPUT: outputs}

    def predict(self, chars):
        # batch_size, max_len = words.size()
        chars = self.embedding(chars)
        outputs = self.mlp(chars)

        return {Const.OUTPUT: outputs}

embed = BertEmbedding(data.get_vocab(Const.CHAR_INPUT), model_dir_or_name='cn-wwm-ext',
                        pool_method='first', requires_grad=True, layers='11', include_cls_sep=False, dropout=0.5)

callbacks = [
                GradientClipCallback(clip_type='norm', clip_value=1),
                WarmupCallback(warmup=0.1, schedule='linear')
            ]

model = BertCNNER(embed, len(data.vocabs[Const.TARGET]))
optimizer = AdamW(model.parameters(), lr=3e-5)

for name, dataset in data.datasets.items():
    original_len = len(dataset)
    dataset.drop(lambda x:x['seq_len']>256, inplace=True)
    clipped_len = len(dataset)
    print("Delete {} instances in {}.".format(original_len-clipped_len, name))

trainer = Trainer(train_data=data.datasets['train'], model=model, optimizer=optimizer, sampler=BucketSampler(),
                  device=0, dev_data=data.datasets['test'], batch_size=6,
                  metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type),
                  loss=CrossEntropyLoss(reduction='sum'),
                  callbacks=callbacks, num_workers=2, n_epochs=5,
                  check_code_level=0, update_every=3)
trainer.train()

