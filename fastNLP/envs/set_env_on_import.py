# 本文件主要用于在分布式启动的情况下，各个backend应该可以提前确定 FASTNLP_GLOBAL_RANK（例如根据环境变量的中RANK值）。
# 注意！仅有当确定我们的训练是分布式训练时，我们才会将 FASTNLP_GLOBAL_RANK 注入到环境变量中；
import os
import sys
from .env import *
import datetime

__all__ = []

def remove_local_rank_in_argv():
    """
    通过 torch.distributed.launch 启动的时候，如果没有加入参数 --use_env ，pytorch 会默认通过 rank 注入 rank，这就
    要求代码中必须有能够 parse rank 的parser，这里将 rank 删除掉，防止后续报错。

    :return:
    """
    index = -1
    for i, v in enumerate(sys.argv):
        if v.startswith('--local_rank='):
            os.environ['LOCAL_RANK'] = v.split('=')[1]
            index = i
            break
    if index != -1:
        sys.argv.pop(index)


def set_env_on_import_torch():
    if 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ and 'RANK' in os.environ:
        os.environ[FASTNLP_GLOBAL_RANK] = os.environ['RANK']
        if int(os.environ.get(FASTNLP_REMOVE_LOCAL_RANK, 1)):
            remove_local_rank_in_argv()

    if 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ and 'RANK' in os.environ and \
            FASTNLP_DISTRIBUTED_CHECK not in os.environ:
        os.environ[FASTNLP_BACKEND_LAUNCH] = '1'


# TODO paddle may need set this
def set_env_on_import_paddle():
    if "PADDLE_TRAINERS_NUM" in os.environ and "PADDLE_TRAINER_ID" in os.environ \
        and "PADDLE_RANK_IN_NODE" in os.environ:
        # 检测到了分布式环境的环境变量
        os.environ[FASTNLP_GLOBAL_RANK] = os.environ["PADDLE_TRAINER_ID"]
        # 如果不是由 fastnlp 启动的
        if FASTNLP_DISTRIBUTED_CHECK not in os.environ:
            os.environ[FASTNLP_BACKEND_LAUNCH] = "1"

# TODO jittor may need set this
def set_env_on_import_jittor():
    # todo 需要设置 FASTNLP_GLOBAL_RANK 和 FASTNLP_BACKEND_LAUNCH
    if 'log_silent' not in os.environ:
        os.environ['log_silent'] = '1'

def set_env_on_import_oneflow():
    if 'GLOG_log_dir' in os.environ:
        os.environ[FASTNLP_GLOBAL_RANK] = os.environ['RANK']
        if int(os.environ.get(FASTNLP_REMOVE_LOCAL_RANK, 1)):
            remove_local_rank_in_argv()

    if 'GLOG_log_dir' in os.environ and FASTNLP_DISTRIBUTED_CHECK not in os.environ:
        os.environ[FASTNLP_BACKEND_LAUNCH] = '1'


def set_env_on_import():
    """
    设置环境变量

    :return:
    """
    # 框架相关的变量设置
    set_env_on_import_torch()
    set_env_on_import_paddle()
    set_env_on_import_jittor()
    set_env_on_import_oneflow()

    # fastNLP 内部使用的一些变量
    if FASTNLP_LAUNCH_TIME not in os.environ:
        cur_time = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')}"
        os.environ[FASTNLP_LAUNCH_TIME] = cur_time

    # 设置对应的值
    if FASTNLP_LOG_LEVEL not in os.environ:
        os.environ[FASTNLP_LOG_LEVEL] = 'AUTO'
