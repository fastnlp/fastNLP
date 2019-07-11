=====================
Metric 教程
=====================

在进行训练时，fastNLP提供了各种各样的 :mod:`~fastNLP.core.metrics` 。
如 :doc:`/user/quickstart`  中所介绍的，:class:`~fastNLP.AccuracyMetric` 类的对象被直接传到 :class:`~fastNLP.Trainer` 中用于训练

.. code-block:: python

    from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric

    trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data,
                      loss=CrossEntropyLoss(), metrics=AccuracyMetric())
    trainer.train()

除了 :class:`~fastNLP.AccuracyMetric` 之外，:class:`~fastNLP.SpanFPreRecMetric` 也是一种非常见的评价指标，
例如在序列标注问题中，常以span的方式计算 F-measure, precision, recall。

另外，fastNLP 还实现了用于抽取式QA（如SQuAD）的metric :class:`~fastNLP.ExtractiveQAMetric`。
用户可以参考下面这个表格，点击第一列查看各个 :mod:`~fastNLP.core.metrics` 的详细文档。

.. csv-table::
   :header: 名称, 介绍

   :class:`~fastNLP.core.metrics.MetricBase` , 自定义metrics需继承的基类
   :class:`~fastNLP.core.metrics.AccuracyMetric` , 简单的正确率metric
   :class:`~fastNLP.core.metrics.SpanFPreRecMetric` , "同时计算 F-measure, precision, recall 值的 metric"
   :class:`~fastNLP.core.metrics.ExtractiveQAMetric` , 用于抽取式QA任务 的metric

更多的 :mod:`~fastNLP.core.metrics` 正在被添加到 fastNLP 当中，敬请期待。

------------------------------
定义自己的metrics
------------------------------

在定义自己的metrics类时需继承 fastNLP 的 :class:`~fastNLP.core.metrics.MetricBase`,
并覆盖写入 ``evaluate`` 和 ``get_metric`` 方法。

    evaluate(xxx) 中传入一个批次的数据，将针对一个批次的预测结果做评价指标的累计

    get_metric(xxx) 当所有数据处理完毕时调用该方法，它将根据 evaluate函数累计的评价指标统计量来计算最终的评价结果

以分类问题中，Accuracy计算为例，假设model的forward返回dict中包含 `pred` 这个key, 并且该key需要用于Accuracy::

    class Model(nn.Module):
        def __init__(xxx):
            # do something
        def forward(self, xxx):
            # do something
            return {'pred': pred, 'other_keys':xxx} # pred's shape: batch_size x num_classes

假设dataset中 `label` 这个field是需要预测的值，并且该field被设置为了target
对应的AccMetric可以按如下的定义, version1, 只使用这一次::

    class AccMetric(MetricBase):
        def __init__(self):
            super().__init__()

            # 根据你的情况自定义指标
            self.corr_num = 0
            self.total = 0

        def evaluate(self, label, pred): # 这里的名称需要和dataset中target field与model返回的key是一样的，不然找不到对应的value
            # dev或test时，每个batch结束会调用一次该方法，需要实现如何根据每个batch累加metric
            self.total += label.size(0)
            self.corr_num += label.eq(pred).sum().item()

        def get_metric(self, reset=True): # 在这里定义如何计算metric
            acc = self.corr_num/self.total
            if reset: # 是否清零以便重新计算
                self.corr_num = 0
                self.total = 0
            return {'acc': acc} # 需要返回一个dict，key为该metric的名称，该名称会显示到Trainer的progress bar中


version2，如果需要复用Metric，比如下一次使用AccMetric时，dataset中目标field不叫label而叫y，或者model的输出不是pred::

    class AccMetric(MetricBase):
        def __init__(self, label=None, pred=None):
            # 假设在另一场景使用时，目标field叫y，model给出的key为pred_y。则只需要在初始化AccMetric时，
            #   acc_metric = AccMetric(label='y', pred='pred_y')即可。
            # 当初始化为acc_metric = AccMetric()，即label=None, pred=None, fastNLP会直接使用'label', 'pred'作为key去索取对
            #   应的的值
            super().__init__()
            self._init_param_map(label=label, pred=pred) # 该方法会注册label和pred. 仅需要注册evaluate()方法会用到的参数名即可
            # 如果没有注册该则效果与version1就是一样的

            # 根据你的情况自定义指标
            self.corr_num = 0
            self.total = 0

        def evaluate(self, label, pred): # 这里的参数名称需要和self._init_param_map()注册时一致。
            # dev或test时，每个batch结束会调用一次该方法，需要实现如何根据每个batch累加metric
            self.total += label.size(0)
            self.corr_num += label.eq(pred).sum().item()

        def get_metric(self, reset=True): # 在这里定义如何计算metric
            acc = self.corr_num/self.total
            if reset: # 是否清零以便重新计算
                self.corr_num = 0
                self.total = 0
            return {'acc': acc} # 需要返回一个dict，key为该metric的名称，该名称会显示到Trainer的progress bar中


``MetricBase`` 将会在输入的字典 ``pred_dict`` 和 ``target_dict`` 中进行检查.
``pred_dict`` 是模型当中 ``forward()`` 函数或者 ``predict()`` 函数的返回值.
``target_dict`` 是DataSet当中的ground truth, 判定ground truth的条件是field的 ``is_target`` 被设置为True.

``MetricBase`` 会进行以下的类型检测:

1. self.evaluate当中是否有varargs, 这是不支持的.
2. self.evaluate当中所需要的参数是否既不在 ``pred_dict`` 也不在 ``target_dict`` .
3. self.evaluate当中所需要的参数是否既在 ``pred_dict`` 也在 ``target_dict`` .

除此以外，在参数被传入self.evaluate以前，这个函数会检测 ``pred_dict`` 和 ``target_dict`` 当中没有被用到的参数
如果kwargs是self.evaluate的参数，则不会检测


self.evaluate将计算一个批次(batch)的评价指标，并累计。 没有返回值
self.get_metric将统计当前的评价指标并返回评价结果, 返回值需要是一个dict, key是指标名称，value是指标的值

