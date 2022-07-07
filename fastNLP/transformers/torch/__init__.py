"""
为了防止因 `transformers <https://github.com/huggingface/transformers>`_ 版本变化导致代码不兼容，当前文件夹以及子文件夹
都复制自 `transformers <https://github.com/huggingface/transformers>`_  的 4.11.3 版本。

In order to avoid the code change of `transformers <https://github.com/huggingface/transformers>`_ to cause version
mismatch, we copy code from `transformers <https://github.com/huggingface/transformers>`_ (version:4.11.3) in this
folder and its subfolder.

您可以如下面代码所示使用 transformers::

    from fastNLP.transformers.torch import BertModel
    ...
"""
__version__ = "4.11.3"
from .models import *