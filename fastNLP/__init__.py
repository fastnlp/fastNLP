"""
fastNLP 由 :mod:`~fastNLP.core` 、 :mod:`~fastNLP.io` 、:mod:`~fastNLP.modules` 等子模块组成，但常用的组件都可以直接 import ，常用组件如下：
"""
__all__ = ["Instance", "FieldArray", "Batch", "Vocabulary", "DataSet",
           "Trainer", "Tester", "Callback",
           "Padder", "AutoPadder", "EngChar2DPadder",
           "AccuracyMetric", "Optimizer", "SGD", "Adam",
           "Sampler", "SequentialSampler", "BucketSampler", "RandomSampler",
           "LossFunc", "CrossEntropyLoss", "L1Loss", "BCELoss", "NLLLoss", "LossInForward",
           "cache_results"]
from .core import *
from . import models
from . import modules

__version__ = '0.4.0'
