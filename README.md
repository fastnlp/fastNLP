# fastNLP


[//]: # ([![Build Status]&#40;https://travis-ci.org/fastnlp/fastNLP.svg?branch=master&#41;]&#40;https://travis-ci.org/fastnlp/fastNLP&#41;)

[//]: # ([![codecov]&#40;https://codecov.io/gh/fastnlp/fastNLP/branch/master/graph/badge.svg&#41;]&#40;https://codecov.io/gh/fastnlp/fastNLP&#41;)

[//]: # ([![Pypi]&#40;https://img.shields.io/pypi/v/fastNLP.svg&#41;]&#40;https://pypi.org/project/fastNLP&#41;)

[//]: # (![Hex.pm]&#40;https://img.shields.io/hexpm/l/plug.svg&#41;)

[//]: # ([![Documentation Status]&#40;https://readthedocs.org/projects/fastnlp/badge/?version=latest&#41;]&#40;http://fastnlp.readthedocs.io/?badge=latest&#41;)


fastNLP是一款轻量级的自然语言处理（NLP）工具包，目标是减少用户项目中的工程型代码，例如数据处理循环、训练循环、多卡运行等。

fastNLP具有如下的特性：

- 便捷。在数据处理中可以通过apply函数避免循环、使用多进程提速等；在训练循环阶段可以很方便定制操作。
- 高效。无需改动代码，实现fp16切换、多卡、ZeRO优化等。
- 兼容。fastNLP支持多种深度学习框架作为后端。

> :warning: **为了实现对不同深度学习架构的兼容，fastNLP 1.0.0之后的版本重新设计了架构，因此与过去的fastNLP版本不完全兼容，
> 基于更早的fastNLP代码需要做一定的调整**: 

## fastNLP文档
[中文文档](https://fastnlp.readthedocs.io/)

## 安装指南
fastNLP可以通过以下的命令进行安装
```shell
pip install fastNLP
```
如果需要安装更早版本的fastNLP请指定版本号，例如
```shell
pip install fastNLP==0.7.1
```
另外，请根据使用的深度学习框架，安装相应的深度学习框架。

<details>
<summary>Pytorch</summary>
下面是使用pytorch来进行文本分类的例子。需要安装torch>=1.6.0。

```python
from fastNLP.io import ChnSentiCorpLoader
from functools import partial
from fastNLP import cache_results
from fastNLP.transformers.torch import BertTokenizer

# 使用cache_results装饰器装饰函数，将prepare_data的返回结果缓存到caches/cache.pkl，再次运行时，如果
#  该文件还存在，将自动读取缓存文件，而不再次运行预处理代码。
@cache_results('caches/cache.pkl')
def prepare_data():
    # 会自动下载数据，并且可以通过文档看到返回的 dataset 应该是包含"raw_words"和"target"两个field的
    data_bundle = ChnSentiCorpLoader().load()
    # 使用tokenizer对数据进行tokenize
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')
    tokenize = partial(tokenizer, max_length=256)  # 限制数据的最大长度
    data_bundle.apply_field_more(tokenize, field_name='raw_chars', num_proc=4)  # 会新增"input_ids", "attention_mask"等field进入dataset中
    data_bundle.apply_field(int, field_name='target', new_field_name='labels')  # 将int函数应用到每个target上，并且放入新的labels field中
    return data_bundle
data_bundle = prepare_data()
print(data_bundle.get_dataset('train')[:4])

# 初始化model, optimizer
from fastNLP.transformers.torch import BertForSequenceClassification
from torch import optim
model = BertForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm')
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# 准备dataloader
from fastNLP import prepare_dataloader
dls = prepare_dataloader(data_bundle, batch_size=32)

# 准备训练
from fastNLP import Trainer, Accuracy, LoadBestModelCallback, TorchWarmupCallback, Event
callbacks = [
    TorchWarmupCallback(warmup=0.1, schedule='linear'),   # 训练过程中调整学习率。
    LoadBestModelCallback()  # 将在训练结束之后，加载性能最优的model
]
# 在训练特定时机加入一些操作， 不同时机能够获取到的参数不一样，可以通过Trainer.on函数的文档查看每个时机的参数
@Trainer.on(Event.on_before_backward())
def print_loss(trainer, outputs):
    if trainer.global_forward_batches % 10 == 0:  # 每10个batch打印一次loss。
        print(outputs.loss.item())

trainer = Trainer(model=model, train_dataloader=dls['train'], optimizers=optimizer,
                  device=0, evaluate_dataloaders=dls['dev'], metrics={'acc': Accuracy()},
                  callbacks=callbacks, monitor='acc#acc',n_epochs=5,
                  # Accuracy的update()函数需要pred，target两个参数，它们实际对应的就是以下的field。
                  evaluate_input_mapping={'labels': 'target'},  # 在评测时，将dataloader中会输入到模型的labels重新命名为target
                  evaluate_output_mapping={'logits': 'pred'}  # 在评测时，将model输出中的logits重新命名为pred
                  )
trainer.run()

# 在测试集合上进行评测
from fastNLP import Evaluator
evaluator = Evaluator(model=model, dataloaders=dls['test'], metrics={'acc': Accuracy()},
                      # Accuracy的update()函数需要pred，target两个参数，它们实际对应的就是以下的field。
                      output_mapping={'logits': 'pred'},
                      input_mapping={'labels': 'target'})
evaluator.run()
```

更多内容可以参考如下的链接
### 快速入门

- [0. 10 分钟快速上手 fastNLP torch](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_torch_tutorial.html)

### 详细使用教程

- [1. Trainer 和 Evaluator 的基本使用](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_tutorial_0.html)
- [2. DataSet 和 Vocabulary 的基本使用](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_tutorial_1.html)
- [3. DataBundle 和 Tokenizer 的基本使用](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_tutorial_2.html)
- [4. TorchDataloader 的内部结构和基本使用](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_tutorial_3.html)
- [5. fastNLP 中的预定义模型](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_tutorial_4.html)
- [6. Trainer 和 Evaluator 的深入介绍](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_tutorial_4.html)
- [7. fastNLP 与 paddle 或 jittor 的结合](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_tutorial_5.html)
- [8. 使用 Bert + fine-tuning 完成 SST-2 分类](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_tutorial_e1.html)
- [9. 使用 Bert + prompt 完成 SST-2 分类](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_tutorial_e2.html)


</details>

<details>
<summary>Paddle</summary>
下面是使用paddle来进行文本分类的例子。需要安装paddle>=2.2.0以及paddlenlp>=2.3.3。

```python
from fastNLP.io import ChnSentiCorpLoader
from functools import partial

# 会自动下载数据，并且可以通过文档看到返回的 dataset 应该是包含"raw_words"和"target"两个field的
data_bundle = ChnSentiCorpLoader().load()

# 使用tokenizer对数据进行tokenize
from paddlenlp.transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')
tokenize = partial(tokenizer, max_length=256)  # 限制一下最大长度
data_bundle.apply_field_more(tokenize, field_name='raw_chars', num_proc=4)  # 会新增"input_ids", "attention_mask"等field进入dataset中
data_bundle.apply_field(int, field_name='target', new_field_name='labels')  # 将int函数应用到每个target上，并且放入新的labels field中
print(data_bundle.get_dataset('train')[:4])

# 初始化 model 
from paddlenlp.transformers import BertForSequenceClassification, LinearDecayWithWarmup
from paddle import optimizer, nn
class SeqClsModel(nn.Layer):
    def __init__(self, model_checkpoint, num_labels):
        super(SeqClsModel, self).__init__()
        self.num_labels = num_labels
        self.bert = BertForSequenceClassification.from_pretrained(model_checkpoint)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        logits = self.bert(input_ids, token_type_ids, position_ids, attention_mask)
        return logits

    def train_step(self, input_ids, labels, token_type_ids=None, position_ids=None, attention_mask=None):
        logits = self(input_ids, token_type_ids, position_ids, attention_mask)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1, )))
        return {
            "logits": logits,
            "loss": loss,
        }
    
    def evaluate_step(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        logits = self(input_ids, token_type_ids, position_ids, attention_mask)
        return {
            "logits": logits,
        }

model = SeqClsModel('hfl/chinese-bert-wwm', num_labels=2)

# 准备dataloader
from fastNLP import prepare_dataloader
dls = prepare_dataloader(data_bundle, batch_size=16)

# 训练过程中调整学习率。
scheduler = LinearDecayWithWarmup(2e-5, total_steps=20 * len(dls['train']), warmup=0.1)
optimizer = optimizer.AdamW(parameters=model.parameters(), learning_rate=scheduler)

# 准备训练
from fastNLP import Trainer, Accuracy, LoadBestModelCallback, Event
callbacks = [
    LoadBestModelCallback()  # 将在训练结束之后，加载性能最优的model
]
# 在训练特定时机加入一些操作， 不同时机能够获取到的参数不一样，可以通过Trainer.on函数的文档查看每个时机的参数
@Trainer.on(Event.on_before_backward())
def print_loss(trainer, outputs):
    if trainer.global_forward_batches % 10 == 0:  # 每10个batch打印一次loss。
        print(outputs["loss"].item())

trainer = Trainer(model=model, train_dataloader=dls['train'], optimizers=optimizer,
                  device=0, evaluate_dataloaders=dls['dev'], metrics={'acc': Accuracy()},
                  callbacks=callbacks, monitor='acc#acc',
                  # Accuracy的update()函数需要pred，target两个参数，它们实际对应的就是以下的field。
                  evaluate_output_mapping={'logits': 'pred'},
                  evaluate_input_mapping={'labels': 'target'}
                  )
trainer.run()

# 在测试集合上进行评测
from fastNLP import Evaluator
evaluator = Evaluator(model=model, dataloaders=dls['test'], metrics={'acc': Accuracy()},
                      # Accuracy的update()函数需要pred，target两个参数，它们实际对应的就是以下的field。
                      output_mapping={'logits': 'pred'},
                      input_mapping={'labels': 'target'})
evaluator.run()
```

更多内容可以参考如下的链接
### 快速入门

- [0. 10 分钟快速上手 fastNLP paddle](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_torch_tutorial.html)

### 详细使用教程

- [1. 使用 paddlenlp 和 fastNLP 实现中文文本情感分析](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_tutorial_paddle_e1.html)
- [2. 使用 paddlenlp 和 fastNLP 训练中文阅读理解任务](http://www.fastnlp.top/docs/fastNLP/master/tutorials/fastnlp_tutorial_paddle_e2.html)

</details>

<details>
<summary>oneflow</summary>
</details>



<details>
<summary>jittor</summary>
</details>


## 项目结构

fastNLP的项目结构如下：

<table>
<tr>
    <td><b> fastNLP </b></td>
    <td> 开源的自然语言处理库 </td>
</tr>
<tr>
    <td><b> fastNLP.core </b></td>
    <td> 实现了核心功能，包括数据处理组件、训练器、测试器等 </td>
</tr>
<tr>
    <td><b> fastNLP.models </b></td>
    <td> 实现了一些完整的神经网络模型 </td>
</tr>
<tr>
    <td><b> fastNLP.modules </b></td>
    <td> 实现了用于搭建神经网络模型的诸多组件 </td>
</tr>
<tr>
    <td><b> fastNLP.embeddings </b></td>
    <td> 实现了将序列index转为向量序列的功能，包括读取预训练embedding等 </td>
</tr>
<tr>
    <td><b> fastNLP.io </b></td>
    <td> 实现了读写功能，包括数据读入与预处理，模型读写，数据与模型自动下载等 </td>
</tr>
</table>

<hr>

