"""
fastNLP 由 :mod:`~fastNLP.core` 、 :mod:`~fastNLP.io` 、:mod:`~fastNLP.modules`、:mod:`~fastNLP.models`
和 :mod:`~fastNLP.component` 等子模块组成。

- :mod:`~fastNLP.core` fastNLP 的核心模块，包括 DataSet、 Trainer、 Tester 等组件
- :mod:`~fastNLP.io` fastNLP 的输入输出模块，实现了数据集的读取，模型的存取等功能
- :mod:`~fastNLP.modules`  TODO 如何介绍
- :mod:`~fastNLP.models` 使用 fastNLP 实现的一些常见模型，具体参见 :doc:`fastNLP.models`
- :mod:`~fastNLP.component` TODO

fastNLP 中最常用的组件可以直接从 fastNLP 包中 import ，他们的文档如下：
"""
__all__ = [
    "Instance",
    "FieldArray",
    "Batch",
    "Vocabulary",
    "DataSet",
    "Const",
    
    "Trainer",
    "Tester",
    
    "Callback",
    "GradientClipCallback",
    "EarlyStopCallback",
    "TensorboardCallback",
    "LRScheduler",
    "ControlC",
    
    "Padder",
    "AutoPadder",
    "EngChar2DPadder",
    
    "AccuracyMetric",
    "SpanFPreRecMetric",
    "SQuADMetric",
    
    "Optimizer",
    "SGD",
    "Adam",
    
    "Sampler",
    "SequentialSampler",
    "BucketSampler",
    "RandomSampler",
    
    "LossFunc",
    "CrossEntropyLoss",
    "L1Loss", "BCELoss",
    "NLLLoss",
    "LossInForward",
    
    "cache_results"
]
from .core import *
from . import models
from . import modules

__version__ = '0.4.0'
