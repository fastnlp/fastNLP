"""
为了防止因 https://github.com/huggingface/transformers 版本变化导致代码不兼容，当前 folder 以及子 folder 
都复制自 https://github.com/huggingface/transformers 的4.11.3版本。
In order to avoid the code change of https://github.com/huggingface/transformers to cause version
mismatch, we copy code from https://github.com/huggingface/transformers(version:4.11.3) in this
folder and its subfolder.
"""
__version__ = "4.11.3"
from .models import *