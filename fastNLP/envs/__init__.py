r"""

"""
__all__ = [
    'dump_fastnlp_backend',
    'is_cur_env_distributed',
    'get_global_rank',
    'rank_zero_call',
    'all_rank_call_context',
    'get_gpu_count',
    'fastnlp_no_sync_context'
]


from .env import *
from .set_env_on_import import set_env_on_import
# 首先保证 FASTNLP_GLOBAL_RANK 正确设置
set_env_on_import()
from .set_backend import dump_fastnlp_backend, _set_backend
# 再设置 backend 相关
_set_backend()
from .imports import *
from .utils import _module_available, get_gpu_count
from .distributed import *
