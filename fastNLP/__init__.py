
# 首先保证 FASTNLP_GLOBAL_RANK 正确设置
from fastNLP.envs.set_env_on_import import set_env_on_import
set_env_on_import()

# 再设置 backend 相关
from fastNLP.envs.set_backend import _set_backend
_set_backend()

from fastNLP.core import Trainer, Evaluator