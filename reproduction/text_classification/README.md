# text_classification任务模型复现
这里使用fastNLP复现以下模型：

char_cnn :论文链接[Character-level Convolutional Networks for Text Classiﬁcation](https://arxiv.org/pdf/1509.01626v3.pdf)

dpcnn:论文链接[Deep Pyramid Convolutional Neural Networks for TextCategorization](https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf)

HAN:论文链接[Hierarchical Attention Networks for Document Classiﬁcation](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

LSTM+self_attention:论文链接[A Structured Self-attentive Sentence Embedding](<https://arxiv.org/pdf/1703.03130.pdf>)

AWD-LSTM:论文链接[Regularizing and Optimizing LSTM Language Models](<https://arxiv.org/pdf/1708.02182.pdf>)

# 数据集及复现结果汇总

使用fastNLP复现的结果vs论文汇报结果(/前为fastNLP实现，后面为论文报道,-表示论文没有在该数据集上列出结果)

model name | yelp_p | yelp_f | sst-2|IMDB
:---: | :---: | :---: | :---: |----- 
char_cnn | 93.80/95.12 | - | - |- 
dpcnn | 95.50/97.36 | - | - |- 
HAN |- | - | - |-
LSTM| 95.74/- |- |- |88.52/-
AWD-LSTM| 95.96/- |- |- |88.91/-
LSTM+self_attention| 96.34/- | - | - |89.53/-

