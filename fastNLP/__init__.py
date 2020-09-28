r"""
fastNLP 由 :mod:`~fastNLP.core` 、 :mod:`~fastNLP.io` 、:mod:`~fastNLP.embeddings` 、 :mod:`~fastNLP.modules`、
:mod:`~fastNLP.models` 等子模块组成，你可以查看每个模块的文档。

- :mod:`~fastNLP.core` 是fastNLP 的核心模块，包括 DataSet、 Trainer、 Tester 等组件。详见文档 :mod:`fastNLP.core`
- :mod:`~fastNLP.io` 是实现输入输出的模块，包括了数据集的读取，模型的存取等功能。详见文档 :mod:`fastNLP.io`
- :mod:`~fastNLP.embeddings` 提供用于构建复杂网络模型所需的各种embedding。详见文档 :mod:`fastNLP.embeddings`
- :mod:`~fastNLP.modules`  包含了用于搭建神经网络模型的诸多组件，可以帮助用户快速搭建自己所需的网络。详见文档 :mod:`fastNLP.modules`
- :mod:`~fastNLP.models` 包含了一些使用 fastNLP 实现的完整网络模型，包括 :class:`~fastNLP.models.CNNText` 、 :class:`~fastNLP.models.SeqLabeling` 等常见模型。详见文档 :mod:`fastNLP.models`

fastNLP 中最常用的组件可以直接从 fastNLP 包中 import ，他们的文档如下：
"""
__all__ = [
    "Instance",
    "FieldArray",
    
    "DataSetIter",
    "BatchIter",
    "TorchLoaderIter",
    
    "Vocabulary",
    "DataSet",
    "Const",
    
    "Trainer",
    "Tester",

    "DistTrainer",
    "get_local_rank",
    
    "Callback",
    "GradientClipCallback",
    "EarlyStopCallback",
    "FitlogCallback",
    "EvaluateCallback",
    "LRScheduler",
    "ControlC",
    "LRFinder",
    "TensorboardCallback",
    "WarmupCallback",
    'SaveModelCallback',
    "CallbackException",
    "EarlyStopError",
    "CheckPointCallback",

    "Padder",
    "AutoPadder",
    "EngChar2DPadder",

    # "CollateFn",
    "ConcatCollateFn",

    "MetricBase",
    "AccuracyMetric",
    "SpanFPreRecMetric",
    "CMRC2018Metric",
    "ClassifyFPreRecMetric",
    "ConfusionMatrixMetric",
    
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    
    "Sampler",
    "SequentialSampler",
    "BucketSampler",
    "RandomSampler",
    "SortedSampler",
    "ConstantTokenNumSampler",
    
    "LossFunc",
    "CrossEntropyLoss",
    "MSELoss",
    "L1Loss",
    "BCELoss",
    "NLLLoss",
    "LossInForward",
    "LossBase",
    "CMRC2018Loss",
    
    "cache_results",
    
    'logger',
    "init_logger_dist",
]
__version__ = '0.5.6'

import sys

from . import embeddings
from . import models
from . import modules
from .core import *
from .doc_utils import doc_process
from .io import loader, pipe

doc_process(sys.modules[__name__])
