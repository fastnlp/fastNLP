# Summarization 

## Extractive Summarization


### Models

FastNLP中实现的模型包括：

1. Get To The Point: Summarization with Pointer-Generator Networks (See et al. 2017)
2. Searching for Effective Neural Extractive Summarization  What Works and What's Next (Zhong et al. 2019)
3. Fine-tune BERT for Extractive Summarization (Liu et al. 2019)




### Dataset

这里提供的摘要任务数据集包括：

- CNN/DailyMail
- Newsroom
- The New York Times Annotated Corpus
    - NYT
    - NYT50
- DUC
    - 2002 Task4
    - 2003/2004 Task1
- arXiv
- PubMed


其中公开数据集(CNN/DailyMail, Newsroom, arXiv, PubMed)预处理之后的下载地址：

- [百度云盘](https://pan.baidu.com/s/11qWnDjK9lb33mFZ9vuYlzA) (提取码：h1px)
- [Google Drive](https://drive.google.com/file/d/1uzeSdcLk5ilHaUTeJRNrf-_j59CQGe6r/view?usp=drivesdk)

未公开数据集(NYT, NYT50, DUC)数据处理部分脚本放置于data文件夹



### Evaluation

#### FastRougeMetric

FastRougeMetric使用python实现的ROUGE非官方库来实现在训练过程中快速计算rouge近似值。
 源代码可见 [https://github.com/pltrdy/rouge](https://github.com/pltrdy/rouge)

在fastNLP中，该方法已经被包装成Metric.py中的FastRougeMetric类以供trainer直接使用。
需要事先使用pip安装该rouge库。

    pip install rouge


**注意：由于实现细节的差异，该结果和官方ROUGE结果存在1-2个点的差异，仅可作为训练过程优化趋势的粗略估计。**

    

#### PyRougeMetric

PyRougeMetric 使用论文 [*ROUGE: A Package for Automatic Evaluation of Summaries*](https://www.aclweb.org/anthology/W04-1013) 提供的官方ROUGE 1.5.5评测库。

由于原本的ROUGE使用perl解释器，[pyrouge](https://github.com/bheinzerling/pyrouge)对其进行了python包装，而PyRougeMetric将其进一步包装为trainer可以直接使用的Metric类。

为了使用ROUGE 1.5.5，需要使用sudo权限安装一系列依赖库。

1. ROUGE 本身在Ubuntu下的安装可以参考[博客](https://blog.csdn.net/Hay54/article/details/78744912)
2. 配置wordnet可参考：
```shell
$ cd ~/rouge/RELEASE-1.5.5/data/WordNet-2.0-Exceptions/
$ ./buildExeptionDB.pl . exc WordNet-2.0.exc.db
$ cd ../
$ ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```
3. 安装pyrouge
```shell
$ git clone https://github.com/bheinzerling/pyrouge
$ cd pyrouge
$ python setup.py install
```
4. 测试ROUGE安装是否正确
```shell
$ pyrouge_set_rouge_path /absolute/path/to/ROUGE-1.5.5/directory
$ python -m pyrouge.test
```




### Dataset_loader

- SummarizationLoader: 用于读取处理好的jsonl格式数据集，返回以下field
    - text: 文章正文
    - summary: 摘要
    - domain: 可选，文章发布网站
    - tag: 可选，文章内容标签
    - labels: 抽取式句子标签

- BertSumLoader：用于读取作为 BertSum（Liu 2019） 输入的数据集，返回以下 field：
  - article：每篇文章被截断为 512 后的词表 ID
  - segmet_id：每句话属于 0/1 的 segment
  - cls_id：输入中 ‘[CLS]’ 的位置
  - label：抽取式句子标签



### Train Cmdline

#### Baseline

LSTM + Sequence Labeling

    python train.py --cuda --gpu <gpuid> --sentence_encoder deeplstm --sentence_decoder SeqLab --save_root <savedir> --log_root <logdir> --lr_descent --grad_clip --max_grad_norm 10

Transformer + Sequence Labeling

    python train.py --cuda --gpu <gpuid> --sentence_encoder transformer --sentence_decoder SeqLab --save_root <savedir> --log_root <logdir> --lr_descent --grad_clip --max_grad_norm 10



#### BertSum



### Performance and Hyperparameters

|              Model              | ROUGE-1 | ROUGE-2 | ROUGE-L |                    Paper                    |
| :-----------------------------: | :-----: | :-----: | :-----: | :-----------------------------------------: |
|             LEAD 3              |  40.11  |  17.64  |  36.32  |            our data pre-process             |
|             ORACLE              |  55.24  |  31.14  |  50.96  |            our data pre-process             |
|    LSTM + Sequence Labeling     |  40.72  |  18.27  |  36.98  |                                             |
| Transformer + Sequence Labeling |  40.86  |  18.38  |  37.18  |                                             |
|     LSTM + Pointer Network      |    -    |    -    |    -    |                                             |
|  Transformer + Pointer Network  |    -    |    -    |    -    |                                             |
|             BERTSUM             |  42.71  |  19.76  |  39.03  | Fine-tune BERT for Extractive Summarization |
|         LSTM+PN+BERT+RL         |    -    |    -    |    -    |                                             |



## Abstractive Summarization
Still in Progress...