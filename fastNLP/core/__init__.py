r"""
core 模块里实现了 fastNLP 的核心框架，常用的功能都可以从 fastNLP 包中直接 import。当然你也同样可以从 core 模块的子模块中 import，
例如 :class:`~fastNLP.DataSetIter` 组件有两种 import 的方式::
    
    # 直接从 fastNLP 中 import
    from fastNLP import DataSetIter
    
    # 从 core 模块的子模块 batch 中 import DataSetIter
    from fastNLP.core.batch import DataSetIter

对于常用的功能，你只需要在 :mod:`fastNLP` 中查看即可。如果想了解各个子模块的具体作用，您可以在下面找到每个子模块的具体文档。

"""
__all__ = [
    "DataSet",
    
    "Instance",
    
    "FieldArray",
    "Padder",
    "AutoPadder",
    "EngChar2DPadder",

    "ConcatCollateFn",
    
    "Vocabulary",
    
    "DataSetIter",
    "BatchIter",
    "TorchLoaderIter",
    
    "Const",
    
    "Tester",
    "Trainer",

    "DistTrainer",
    "get_local_rank",

    "cache_results",
    "seq_len_to_mask",
    "get_seq_len",
    "logger",
    "init_logger_dist",

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

    "LossFunc",
    "CrossEntropyLoss",
    "L1Loss",
    "BCELoss",
    "NLLLoss",
    "LossInForward",
    "CMRC2018Loss",
    "MSELoss",
    "LossBase",

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
    
    "SequentialSampler",
    "BucketSampler",
    "RandomSampler",
    "Sampler",
    "SortedSampler",
    "ConstantTokenNumSampler"
]

from ._logger import logger, init_logger_dist
from .batch import DataSetIter, BatchIter, TorchLoaderIter
from .callback import Callback, GradientClipCallback, EarlyStopCallback, FitlogCallback, EvaluateCallback, \
    LRScheduler, ControlC, LRFinder, TensorboardCallback, WarmupCallback, SaveModelCallback, CallbackException, \
    EarlyStopError, CheckPointCallback
from .const import Const
from .dataset import DataSet
from .field import FieldArray, Padder, AutoPadder, EngChar2DPadder
from .instance import Instance
from .losses import LossFunc, CrossEntropyLoss, L1Loss, BCELoss, NLLLoss, \
    LossInForward, CMRC2018Loss, LossBase, MSELoss
from .metrics import AccuracyMetric, SpanFPreRecMetric, CMRC2018Metric, ClassifyFPreRecMetric, MetricBase,\
    ConfusionMatrixMetric
from .optimizer import Optimizer, SGD, Adam, AdamW
from .sampler import SequentialSampler, BucketSampler, RandomSampler, Sampler, SortedSampler, ConstantTokenNumSampler
from .tester import Tester
from .trainer import Trainer
from .utils import cache_results, seq_len_to_mask, get_seq_len
from .vocabulary import Vocabulary
from .collate_fn import ConcatCollateFn
from .dist_trainer import DistTrainer, get_local_rank
