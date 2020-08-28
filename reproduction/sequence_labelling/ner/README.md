# NER任务模型复现
这里使用fastNLP复现经典的BiLSTM-CNN的NER任务的模型，旨在达到与论文中相符的性能。

论文链接[Named Entity Recognition with Bidirectional LSTM-CNNs](https://arxiv.org/pdf/1511.08308.pdf)

# 数据集及复现结果汇总

使用fastNLP复现的结果vs论文汇报结果(/前为fastNLP实现，后面为论文报道)

model name | Conll2003 | Ontonotes 
:---: | :---: | :---: 
BiLSTM-CNN | 91.17/90.91 | 86.47/86.35 |

