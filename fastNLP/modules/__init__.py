"""
modules 模块是 fastNLP 的重要组成部分，它实现了神经网络构建中常见的组件，
具体包括 TODO

可以和 PyTorch 结合使用？TODO

TODO __all__ 里面多暴露一些

"""
from . import aggregator
from . import decoder
from . import encoder
from .aggregator import *
from .decoder import *
from .dropout import TimestepDropout
from .encoder import *
from .utils import get_embeddings

__version__ = '0.0.0'
