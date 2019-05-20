"""
fastNLP 由 :mod:`~fastNLP.core` 、 :mod:`~fastNLP.io` 、:mod:`~fastNLP.modules`、:mod:`~fastNLP.models`
等子模块组成，你可以点进去查看每个模块的文档。

- :mod:`~fastNLP.core` 是fastNLP 的核心模块，包括 DataSet、 Trainer、 Tester 等组件。详见文档 :doc:`/fastNLP.core`
- :mod:`~fastNLP.io` 是实现输入输出的模块，包括了数据集的读取，模型的存取等功能。详见文档 :doc:`/fastNLP.io`
- :mod:`~fastNLP.modules`  包含了用于搭建神经网络模型的诸多组件，可以帮助用户快速搭建自己所需的网络。详见文档 :doc:`/fastNLP.modules`
- :mod:`~fastNLP.models` 包含了一些使用 fastNLP 实现的完整网络模型，包括CNNText、SeqLabeling等常见模型。详见文档 :doc:`/fastNLP.models`

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
__version__ = '0.4.0'

from .core import *
from . import models
from . import modules
