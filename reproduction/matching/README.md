# Matching任务模型复现
这里使用fastNLP复现了几个著名的Matching任务的模型，旨在达到与论文中相符的性能。这几个任务的评价指标均为准确率(%).

复现的模型有（按论文发表时间顺序排序）:
- CNTN：[模型代码](model/cntn.py); [训练代码](matching_cntn.py).
论文链接：[Convolutional Neural Tensor Network Architecture for Community-based Question Answering](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/view/11401/10844). 
- ESIM：[模型代码](model/esim.py); [训练代码](matching_esim.py).
论文链接：[Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038.pdf).
- DIIN：模型代码(still in progress)[](); 训练代码(still in progress)[]().
论文链接：[Natural Language Inference over Interaction Space](https://arxiv.org/pdf/1709.04348.pdf).
- MwAN：[模型代码](model/mwan.py); [训练代码](matching_mwan.py).
论文链接：[Multiway Attention Networks for Modeling Sentence Pairs](https://www.ijcai.org/proceedings/2018/0613.pdf).
- BERT：[模型代码](model/bert.py); [训练代码](matching_bert.py).
论文链接：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf).

# 数据集及复现结果汇总

使用fastNLP复现的结果vs论文汇报结果，在前面的表示使用fastNLP复现的结果

'\-'表示我们仍未复现或者论文原文没有汇报

model name | SNLI | MNLI | RTE | QNLI | Quora
:---: | :---: | :---: | :---: | :---: | :---:
CNTN [代码](model/cntn.py); [论文](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/view/11401/10844) | 77.79 vs - | 63.29/63.16(dev) vs - | 57.04(dev) vs - | 62.38(dev) vs - | - |
ESIM[代码](model/bert.py); [论文](https://arxiv.org/pdf/1609.06038.pdf) | 88.13(glove) vs 88.0(glove)/88.7(elmo) | 77.78/76.49 vs 72.4/72.1* | 59.21(dev) vs - | 76.97(dev) vs - | - |
DIIN [](); [论文](https://arxiv.org/pdf/1709.04348.pdf) | - vs 88.0 | - vs 78.8/77.8 | - | - | - vs 89.06 |
MwAN [代码](model/mwan.py); [论文](https://www.ijcai.org/proceedings/2018/0613.pdf) | 87.9 vs 88.3 | 77.3/76.7(dev) vs 78.5/77.7 | - | 74.6(dev) vs - | 85.6 vs 89.12 |
BERT (BASE version)[代码](model/bert.py); [论文](https://arxiv.org/pdf/1810.04805.pdf) | 90.6 vs - | - vs 84.6/83.4| 67.87(dev) vs 66.4 | 90.97(dev) vs 90.5 | - |

*ESIM模型由MNLI官方复现的结果为72.4/72.1，ESIM原论文当中没有汇报MNLI数据集的结果。

# 数据集复现结果及其他主要模型对比
## SNLI
[Link to SNLI leaderboard](https://nlp.stanford.edu/projects/snli/)

Performance on Test set:

model name | ESIM | DIIN | MwAN | [GPT1.0](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) | [BERT-Large+SRL](https://arxiv.org/pdf/1809.02794.pdf) | [MT-DNN](https://arxiv.org/pdf/1901.11504.pdf) 
:---: | :---: | :---: | :---: | :---: | :---: | :---:
__performance__ | 88.0 | 88.0 | 88.3 | 89.9 | 91.3 | 91.6 |

### 基于fastNLP复现的结果
Performance on Test set:

model name | CNTN | ESIM | DIIN | MwAN | BERT-Base | BERT-Large
:---: | :---: | :---: | :---: | :---: |  :---: | :---:
__performance__ | 77.79 | 88.13 | - | 87.9 | 90.6 | 91.16

## MNLI
[Link to MNLI main page](https://www.nyu.edu/projects/bowman/multinli/)

Performance on Test set(matched/mismatched):

model name | ESIM | DIIN | MwAN | GPT1.0 | BERT-Base | MT-DNN 
:---: | :---: | :---: | :---: | :---: | :---: | :---:
__performance__ | 72.4/72.1 | 78.8/77.8 | 78.5/77.7 | 82.1/81.4 | 84.6/83.4 | 87.9/87.4 |

### 基于fastNLP复现的结果
Performance on Test set(matched/mismatched):

model name | CNTN | ESIM | DIIN | MwAN | BERT-Base 
:---: | :---: | :---: | :---: | :---: | :---: |
__performance__ | 63.29/63.16(dev) | 77.78/76.49 | - | 77.3/76.7(dev) | - |


## RTE

Still in progress.

## QNLI

### From GLUE baselines
[Link to GLUE leaderboard](https://gluebenchmark.com/leaderboard)

Performance on Test set:
#### LSTM-based
model name | BiLSTM | BiLSTM + Attn | BiLSTM + ELMo | BiLSTM + Attn + ELMo
:---: | :---: | :---: | :---: | :---: |
__performance__ | 74.6 | 74.3 | 75.5 | 79.8 |

*这些LSTM-based的baseline是由QNLI官方实现并测试的。

#### Transformer-based
model name | GPT1.0 | BERT-Base | BERT-Large | MT-DNN
:---: | :---: | :---: | :---: | :---: |
__performance__ | 87.4 | 90.5 | 92.7 | 96.0 |



### 基于fastNLP复现的结果
Performance on __Dev__ set:

model name | CNTN | ESIM | DIIN | MwAN | BERT 
:---: | :---: | :---: | :---: | :---: | :---:
__performance__ | 62.38  | 76.97 | - | 74.6 | -

## Quora

Still in progress.

