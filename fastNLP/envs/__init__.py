r"""

"""
__all__ = [
    'dump_fastnlp_backend',

    # utils
    'get_gpu_count',

    # distributed
    "rank_zero_rm",
    'rank_zero_call',
    'get_global_rank',
    'fastnlp_no_sync_context',
    'all_rank_call_context',
    'is_cur_env_distributed',
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
