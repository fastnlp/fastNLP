# fastNLP开发者指南
#### 本教程涉及以下类：
- DataSet
- Sampler
- Batch
- Model
- Loss
- Metric
- Trainer
- Tester

#### DataSet: 用于承载数据。
1. DataSet里面每个元素只能是以下的三类`np.float64`, `np.int64`, `np.str`。如果传入的数据是`int`则被转换为`np.int64`, `float`被转为`np.float64`。   
2. DataSet可以将field设置为input或者target。其中被设置为input的field会被传递给Model.forward, 这个过程中我们是通过键匹配完成传递的。举例来说，假设DataSet中有'x1', 'x2', 'x3'被设置为了input，而
    - 函数是Model.forward(self, x1, x3), 那么DataSet中'x1', 'x3'会被传递给forward函数。多余的'x2'会被忽略
    - 函数是Model.forward(self, x1, x4), 这里多需要了一个'x4', 但是DataSet的input field中没有这个field，会报错。
    - 函数是Model.forward(self, x1, **kwargs), 会把'x1', 'x2', 'x3'都传入。但如果是Model.forward(self, x4, **kwargs)就会发生报错，因为没有'x4'。
3. 对于设置为target的field的名称，我们建议取名为'target'（如果只有一个需要predict的值），但是不强制。后面会讲为什么target可以不强制。  
DataSet应该是不需要单独再开发的，如果有不能满足的场景，请在开发群提出或者github提交issue。

#### Sampler: 给定一个DataSet，返回一个序号的list，Batch按照这个list输出数据。
Sampler需要继承fastNLP.core.sampler.BaseSampler
```python
class BaseSampler(object):
    """The base class of all samplers.

    Sub-classes must implement the __call__ method.
    __call__ takes a DataSet object and returns a list of int - the sampling indices.
    """
def __call__(self, *args, **kwargs):
    raise NotImplementedError
    
# 子类需要复写__call__方法。这个函数只能有一个必选参数, 且必须是DataSet类别， 否则Trainer没法调
class SonSampler(BaseSampler):
    def __init__(self, xxx):
        # 可以实现init也不可以不实现。
        pass
    def __call__(self, data_set):
        pass
```

#### Batch: 将DataSet中设置为input和target的field取出来构成batch_x, batch_y
并且根据情况(主要根据数据类型能不能转为Tensor)将数据转换为pytorch的Tensor。batch中sample的取出顺序是由Sampler决定的。  
Sampler是传入一个DataSet，返回一个与DataSet等长的序号list，Batch一次会取出batch_size个sample(最后一个batch可能数量不足batch_size个)。   
举例：  
1. SequentialSampler是顺序采样

    假设传入的DataSet长度是100, SequentialSampler返回的序号list就是[0, 1, ...,98, 99]. batch_size如果被设置为4，那么第一个batch所获取的instance就是[0, 1, 2, 3]这四个instance. 第二个batch所获取instace就是[4, 5, 6, 7], ...直到采完所有的sample。   
2. RandomSampler是随机采样   

    假设传入的DataSet长度是100, RandomSampler返回的序号list可能是[0, 99, 20, 5, 3, 1, ...]. 依次按照batch_size的大小取出sample。  

Batch应该不需要继承与开发，如果你有特殊需求请在开发群里提出。

#### Model：用户自定的Model
必须是nn.Module的子类
1. 必须实现forward方法，并且forward方法不能出现*arg这种参数. 例如
    ```python
    def forward(self, word_seq, *args): #这是不允许的.
        # ...
        pass
    ```
    返回值必须是dict的  
    ```python
    def forward(self, word_seq, seq_lens):  
        xxx = "xxx"
        return {'pred': xxx} #return的值必须是dict的。里面的预测的key推荐使用pred，但是不做强制限制。输出元素数目不限。  
    ```
2. 如果实现了predict方法，在做evaluation的时候将调用predict方法而不是forward。如果没有predict方法，则在evaluation时调用forward方法。predict方法也不能使用*args这种参数形式，同时结果也必须返回一个dict，同样推荐key为'pred'。

#### Loss: 根据model.forward()返回的prediction(是一个dict)和batch_y计算相应的loss
1. 先介绍"键映射"。 如在DataSet, Model一节所看见的那样，fastNLP并不限制Model.forward()的返回值，也不限制DataSet中target field的key。计算的loss的时候，怎么才能知道从哪里取值呢？  
这里以CrossEntropyLoss为例，一般情况下, 计算CrossEntropy需要prediction和target两个值。而在CrossEntropyLoss初始化时可以传入两个参数(pred=None, target=None), 这两个参数接受的类型是str，假设(pred='output', target='label')，那么CrossEntropyLoss会使用'output'这个key在forward的output与batch_y中寻找值;'label'也是在forward的output与batch_y中寻找值。注意这里pred或target的来源并不一定非要来自于model.forward与batch_y，也可以只来自于forward的结果。  
2. 如何创建一个自己的loss
    - 使用fastNLP.LossInForward, 在model.forward()的结果中包含一个为loss的key。
    - trainer中使用loss(假设loss=CrossEntropyLoss())的时候其实是
        los = loss(prediction, batch_y)，即直接调用的是`loss.__call__()`方法，但是CrossEntropyLoss里面并没有自己实现`__call__`方法，这是因为`__call__`在LossBase中实现了。所有的loss必须继承fastNLP.core.loss.LossBase, 下面先说一下LossBase的几个方法，见下一节。  
3. 尽量不要复写`__call__()`, `_init_param_map()`方法。

```python
class LossBase():
    def __init__(self):
        self.param_map = {} # 一般情况下也不需要自己创建。调用_init_param_map()更好
        self._checked = False # 这个参数可以忽略

    def _init_param_map(self, key_map=None, **kwargs):
        # 这个函数是用于注册Loss的“键映射”，有两种传值方法，
        # 第一种是通过key_map传入dict，取值是用value到forward和batch_y取
        #    key_map = {'pred': 'output', 'target': 'label'}   
        # 第二种是自己写
        #    _init_param_map(pred='output', target='label')
        # 为什么会提供这么一个方法？通过调用这个方法会自动注册param_map，并会做一些检查，防止出现传入的key其实并不是get_loss
        #   的一个参数。注意传入这个方法的参数必须都是需要做键映射的内容，其它loss参数不要传入。如果传入(pred=None, target=None)
        #   则__call__()会到pred_dict与target_dict去寻找key为'pred'和'target'的值。
        # 但这个参数不是必须要调用的。

    def __call__(self, pred_dict, target_dict, check=False): # check=False忽略这个参数，之后应该会被删除的
        # 这个函数主要会做一些check的工作，比如pred_dict与target_dict中是否包含了计算loss所必须的key等。检查通过，则调用get_loss
        #  方法。
        fast_param = self._fast_param_map(predict_dict, target_dict):
        if fast_param:
            return self.get_loss(**fast_param)
        # 如果没有fast_param则通过匹配参数然后调用get_loss完成
        xxxx
        return loss # 返回为Tensor的loss
    def _fast_param_map(self, pred_dict, target_dict):
        # 这是一种快速计算loss的机制，因为在很多情况下其实都不需要通过"键映射"，比如计算loss时，pred_dict只有一个元素，
        #   target_dict也只有一个元素，那么无歧义地就可以把预测值与实际值用于计算loss, 基类判断了这种情况(可能还有其它无歧义的情况)。
        #   即_fast_param_map成功的话，就不需要使用键映射，这样即使在没有传递或者传递错误"键映射"的情况也可以直接计算loss。
        # 返回值是一个dict, 如果匹配成功，应该返回类似{'pred':value, 'target': value}的结果；如果dict为空则说明匹配失败，
        #   __call__方法会继续执行。

    def get_loss(self, *args, **kwargs):
        # 这个是一定需要实现的，计算loss的地方。
        # (1) get_loss中一定不能包含*arg这种参数形式。
        # (2) 如果包含**kwargs这种参数，这会将pred_dict与target_dict中所有参数传入。但是建议不要用这个参数
        raise NotImplementedError

# 下面使用L1Loss举例
class L1Loss(LossBase): # 继承LossBase
    # 初始化需要映射的值，这里需要映射的值'pred', 'target'必须与get_loss需要参数名是对应的
    def __init__(self, pred=None, target=None): 
        super(L1Loss, self).__init__()
        # 这里传入_init_param_map以使得pred和target被正确注册，但这一步不是必须的, 建议调用。传入_init_param_map的是用于
        #   “键映射"的键值对。假设初始化__init__(pred=None, target=None, threshold=0.1)中threshold是用于控制loss计算的，则
        #   不要将threshold传入_init_param_map.
        self._init_param_map(pred=pred, target=target)

    def get_loss(self, pred, target):
        # 这里'pred', 'target'必须和初始化的映射是一致的。
        return F.l1_loss(input=pred, target=target) #直接返回一个loss即可
```

###  Metric: 根据Model.forward()或者Model.predict()的结果计算metric  
metric的设计和loss的设计类似。都是传入pred_dict与target_dict进行计算。但是metric的pred_dict来源可能是Model.forward的返回值， 也可能是Model.predict(如果Model具有predict方法则会调用predict方法)的返回值，下面统一用pred_dict代替。  
1. 这里的"键映射"与loss的"键映射"是类似的。举例来说，若Metric(pred='output', target='label')，则使用'output'到pred_dict和target_dict中寻找pred, 用'label'寻找target。   
2. 如何创建一个自己的Metric方法  
Metric与loss的计算不同在于，Metric的计算有两个步骤。  
    - **每个batch的输出**都会调用Metric的``__call__(pred_dict, target_dict)``方法，而``__call__``方法会调用evaluate()(需要实现)方法。   
    - 在所有batch传入之后，调用Metric的get_metric()方法得到最终的metric值。  
    - 所以Metric在调用evaluate方法时，根据拿到的数据: pred_dict与batch_y, 改变自己的状态(比如累加正确的次数，总的sample数等)。在调用get_metric()的时候给出一个最终计算结果。  
    所有的Metric必须继承自fastNLP.core.metrics.MetricBase. 例子见下一个cell        
3. 尽量不要复写``__call__()``,``_init_param_map()``方法。

```python
class MetricBase: 
    def __init__(self):
        self.param_map = {} # 一般情况下也不需要自己创建。调用_init_param_map()更好
        self._checked = False # 这个参数可以忽略

    def _init_param_map(self, key_map=None, **kwargs):
        # 这个函数是用于注册Metric的“键映射”，有两种传值方法，
        # 第一种是通过key_map传入dict，取值是用value到forward和batch_y取
        #    key_map = {'pred': 'output', 'target': 'label'}   
        # 第二种是自己写(建议使用改种方式)
        #    _init_param_map(pred='output', target='label')
        # 为什么会提供这么一个方法？通过调用这个方法会自动注册param_map，并会做一些检查，防止出现传入的key其实并不是evaluate()
        #   的一个参数。注意传入这个方法的参数必须都是需要做键映射的内容，其它evaluate参数不要传入。如果传入(pred=None, target=None)
        #   则__call__()会到pred_dict与target_dict去寻找key为'pred'和'target'的值。
        # 但这个参数不是必须要调用的。
        pass

    def __call__(self, pred_dict, target_dict, check=False): # check=False忽略这个参数，之后应该会被删除的
        # 这个函数主要会做一些check的工作，比如pred_dict与target_dict中是否包含了计算evaluate所必须的key等。检查通过，则调用
        #  evaluate方法。
        fast_param = self._fast_param_map(predict_dict, target_dict):
        if fast_param:
            return self.evaluate(**fast_param)
        # 如果没有fast_param则通过匹配参数然后调用get_loss完成
        # xxxx

    def _fast_param_map(self, pred_dict, target_dict):
        # 这是一种快速计算loss的机制，因为在很多情况下其实都不需要通过"键映射"，比如evaluate时，pred_dict只有一个元素，
        #   target_dict也只有一个元素，那么无歧义地就可以把预测值与实际值用于计算metric, 基类判断了这种情况(可能还有其它无歧义的
        #   情况)。即_fast_param_map成功的话，就不需要使用键映射，这样即使在没有传递或者传递错误"键映射"的情况也可以直接计算metric。
        # 返回值是一个dict, 如果匹配成功，应该返回类似{'pred':value, 'target': value}的结果；如果dict为空则说明匹配失败，
        #   __call__方法会继续尝试匹配。
        pass

    def evaluate(self, *args, **kwargs):
        # 这个是一定需要实现的，累加metric状态
        # (1) evaluate()中一定不能包含*arg这种参数形式。
        # (2) 如果包含**kwargs这种参数，这会将pred_dict与target_dict中所有参数传入。但是建议不要用这个参数
        raise NotImplementedError

    def get_metric(self, reset=True):
        # 这是一定需要实现的，获取最终的metric。返回值必须是一个dict。会在所有batch传入之后调用
        raise NotImplementedError

# 下面使用AccuracyMetric举例
class AccuracyMetric(MetricBase): # MetricBase
    # 初始化需要映射的值，这里需要映射的值'pred', 'target'必须与evaluate()需要参数名是对应的
    def __init__(self, pred=None, target=None): 
        super(AccuracyMetric, self).__init__()
        # 这里传入_init_param_map以使得pred和target被正确注册，但这一步不是必须的, 建议调用。传入_init_param_map的是用于
        #   “键映射"的键值对。假设初始化__init__(pred=None, target=None, threshold=0.1)中threshold是用于控制loss计算的，则
        #   不要将threshold传入_init_param_map.
        self._init_param_map(pred=pred, target=target)

        self.total = 0 # 用于累加一共有多少sample
        self.corr = 0 # 用于累加一共有多少正确的sample

    def evaluate(self, pred, target):
        # 对pred和target做一些基本的判断或者预处理等
        if pred.size()==target.size() and len(pred.size())=1: #如果pred已经做了argmax
            pass
        elif len(pred.size())==2 and len(target.size())==1: # pred还没有进行argmax
            pred = pred.argmax(dim=1)
        else:
            raise ValueError("The shape of pred and target should be ((B, n_classes), (B, )) or ("
                            "(B,),(B,)).")
        assert pred.size(0)==target.size(0), "Mismatch batch size."
        # 进行相应的累加
        self.total += pred.size(0)
        self.corr += torch.sum(torch.eq(pred, target).float()).item()

    def get_metric(self, reset=True):
        # reset用于指示是否清空累加信息。默认为True
        # 这个函数需要返回dict，可以包含多个metric。
        metric = {}
        metric['acc'] = self.corr/self.total
        if reset:
            self.total = 0
            self.corr = 0
        return metric
```

#### Tester: 用于做evaluation，应该不需要更改
重要的初始化参数有data, model, metric；比较重要的function是test()。

test中的运行过程  
``` 
predict_func = 如果有model.predict则为model.predict, 否则是model.forward  
for batch_x, batch_y in batch:  
# (1) 同步数据与model  
# (2) 根据predict_func的参数从batch_x中取出数据传入到predict_func中，得到结果pred_dict  
# (3) 调用metric(pred_dict, batch_y  
# (4) 当所有batch都运行完毕，会调用metric的get_metric方法，并且以返回的值作为evaluation的结果  
metric.get_metric()
```

#### Trainer: 对训练过程的封装。  
里面比较重要的function是train()  
train()中的运行过程  
```
(1) 创建batch  
    batch = Batch(dataset, batch_size, sampler=sampler)  
    for batch_x, batch_y in batch:  
        # ...
    batch_x，batch_y都是dict。batch_x是DataSet中被设置为input的field；batch_y是DataSet中被设置为target的field。  
    两个dict中的key就是DataSet中的key，value会根据情况做好padding的tensor。
(2)会将batch_x, batch_y中tensor移动到model所在的device  
(3)根据model.forward的参数列表, 从batch_x中取出需要传递给forward的数据。  
(4)获取model.forward的输出结果pred_dict，并与batch_y一起传递给loss函数, 求得loss  
(5)对loss进行反向梯度并更新参数  
(6) 如果有验证集，则需要做验证  
    tester = Tester(model, dev_data，metric)  
    eval_results = tester.test()  
(7) 如果eval_results是当前的最佳结果，则保存模型。  
```

#### 其他
Trainer中还提供了"预跑"的功能。该功能通过check_code_level管理，如果check_code_level为-1，则不进行"预跑"。

check_code_level=0，1，2代表不同的提醒级别。
目前不同提醒级别对应的是对DataSet中设置为input或target但又没有使用的field的提醒级别。
0是忽略(默认)；1是会warning发生了未使用field的情况；2是出现了unused会直接报错并退出运行

"预跑"的主要目的有两个:
- 防止train完了之后进行evaluation的时候出现错误。之前的train就白费了
- 由于存在"键映射"，直接运行导致的报错可能不太容易debug，通过"预跑"过程的报错会有一些debug提示

"预跑"会进行以下的操作：
- 使用很小的batch_size, 检查batch_x中是否包含Model.forward所需要的参数。只会运行两个循环。
- 将Model.foward的输出pred_dict与batch_y输入到loss中， 并尝试backward. 不会更新参数，而且grad会被清零
                  如果传入了dev_data，还将进行metric的测试
- 创建Tester，并传入少量数据，检测是否可以正常运行

"预跑"操作是在Trainer初始化的时候执行的。

正常情况下，应该不需要改动"预跑"的代码。但如果你遇到bug或者有什么好的建议，欢迎在开发群或者github提交issue。


