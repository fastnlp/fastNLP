"""
core 模块里实现了 fastNLP 的核心框架，常用的功能都可以从 fastNLP 包中直接 import。当然你也同样可以从 core 模块的子模块中 import，
例如 Batch 组件有两种 import 的方式::
    
    # 直接从 fastNLP 中 import
    from fastNLP import Batch
    
    # 从 core 模块的子模块 batch 中 import
    from fastNLP.core.batch import Batch

对于常用的功能，你只需要在 :doc:`fastNLP` 中查看即可。如果想了解各个子模块的具体作用，您可以在下面找到每个子模块的具体文档。

.. todo::
    介绍core 的子模块的分工，好像必要性不大
    
"""
from .batch import Batch
from .callback import Callback, GradientClipCallback, EarlyStopCallback, TensorboardCallback, LRScheduler, ControlC
from .const import Const
from .dataset import DataSet
from .field import FieldArray, Padder, AutoPadder, EngChar2DPadder
from .instance import Instance
from .losses import LossFunc, CrossEntropyLoss, L1Loss, BCELoss, NLLLoss, LossInForward
from .metrics import AccuracyMetric, SpanFPreRecMetric, SQuADMetric
from .optimizer import Optimizer, SGD, Adam
from .sampler import SequentialSampler, BucketSampler, RandomSampler, Sampler
from .tester import Tester
from .trainer import Trainer
from .utils import cache_results, seq_len_to_mask
from .vocabulary import Vocabulary
