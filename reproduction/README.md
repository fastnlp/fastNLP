# 模型复现
这里复现了在fastNLP中实现的模型，旨在达到与论文中相符的性能。

复现的模型有:
- Star-Transformer
- ...


## Star-Transformer
[reference](https://arxiv.org/abs/1902.09113)
### Performance (still in progress)
|任务| 数据集 | SOTA | 模型表现 |
|------|------| ------| ------|
|Pos Tagging|CTB 9.0|-|ACC 92.31|
|Pos Tagging|CONLL 2012|-|ACC 96.51|
|Named Entity Recognition|CONLL 2012|-|F1 85.66|
|Text Classification|SST|-|49.18|
|Natural Language Inference|SNLI|-|83.76|

### Usage
``` python
# for sequence labeling(ner, pos tagging, etc)
from fastNLP.models.star_transformer import STSeqLabel
model = STSeqLabel(
    vocab_size=10000, num_cls=50,
    emb_dim=300)


# for sequence classification
from fastNLP.models.star_transformer import STSeqCls
model = STSeqCls(
    vocab_size=10000, num_cls=50,
    emb_dim=300)


# for natural language inference
from fastNLP.models.star_transformer import STNLICls
model = STNLICls(
    vocab_size=10000, num_cls=50,
    emb_dim=300)

```

## ...
