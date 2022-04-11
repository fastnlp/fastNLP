"""
这个文件用于自动以及手动设置某些环境变量的，该文件中的set_env()函数会在 fastNLP 被 import 的时候在set_env_on_import之后运行。可以
    用于设置某些必要的环境变量。同时用户在使用时set_env()修改环境变量时，也应该保证set_env()函数在所有其它代码之前被运行。
"""
import os
import json
import sys
import subprocess
from collections import defaultdict


from fastNLP.envs.env import FASTNLP_BACKEND, FASTNLP_GLOBAL_RANK, USER_CUDA_VISIBLE_DEVICES, FASTNLP_GLOBAL_SEED
from fastNLP.envs.imports import SUPPORT_BACKENDS
from fastNLP.envs.utils import _module_available, get_gpu_count

def _set_backend():
    """
    根据环境变量或者默认配置文件设置 backend 。

     backend 为 paddle 时，我们还将设置部分环境变量以使得 paddle 能够在 fastNLP 中正确运行。
     backend 为 jittor 时，我们将设置 log_silent:1

    :return:
    """
    backend = ''
    if FASTNLP_BACKEND in os.environ:
        backend = os.environ[FASTNLP_BACKEND]
    else:
        # 从文件中读取的
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)
        if conda_env is None:
            conda_env = 'default'
        env_folder = os.path.join(os.path.expanduser('~'), '.fastNLP', 'envs')
        env_path = os.path.join(env_folder, conda_env + '.json')
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r', encoding='utf8') as f:
                    envs = json.load(f)
                    # print(json.dumps(envs))
                if FASTNLP_BACKEND in envs:
                    backend = envs[FASTNLP_BACKEND]
                    os.environ[FASTNLP_BACKEND] = backend
                    if int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
                        print(f"Set fastNLP backend as {backend} based on {env_path}.")
            except BaseException as e:
                raise e

    if backend:
        assert backend in SUPPORT_BACKENDS, f"Right now fastNLP only support the following backends:{SUPPORT_BACKENDS}, " \
                                            f"instead of `{backend}`"

    if backend == 'paddle':
        assert _module_available(backend), f"You must have {backend} available to use {backend} backend."
        assert 'paddle' not in sys.modules, "You have to use `set_backend()` before `import paddle`."
        user_visible_devices = os.getenv(USER_CUDA_VISIBLE_DEVICES)
        if 'PADDLE_RANK_IN_NODE' in os.environ and 'FLAGS_selected_gpus' in os.environ:
            # 在分布式子进程下，根据 USER_VISIBLE_DEVICES 得到进程真正占有的设备
            selected_gpus = os.environ['FLAGS_selected_gpus'].split(',')
            if user_visible_devices is not None:
                # 用户通过 CUDA_VISIBLE_DEVICES 启动了分布式训练
                # 此时经过 set_backend，用户的设置会保存在 USER_CUDA_VISIBLE_DEVICES 中
                # 我们需要从中找到真正使用的设备编号
                user_visible_devices = user_visible_devices.split(",")
                selected_gpus = ",".join([user_visible_devices[int(i)] for i in selected_gpus])
            else:
                # 没有找到 USER_CUDA_VISIBLE_DEVICES，则将之设置为所有的设备
                os.environ[USER_CUDA_VISIBLE_DEVICES] = ",".join(map(str, list(
                    range(get_gpu_count())
                )))
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(selected_gpus)
            os.environ['FLAGS_selected_gpus'] = ",".join([str(g) for g in range(len(selected_gpus))])
            os.environ['FLAGS_selected_accelerators'] = ",".join([str(g) for g in range(len(selected_gpus))])
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            # 主进程中，用户设置了 CUDA_VISIBLE_DEVICES
            # 将用户设置的 CUDA_VISIBLE_DEVICES hack 掉
            CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
            os.environ[USER_CUDA_VISIBLE_DEVICES] = CUDA_VISIBLE_DEVICES
            os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES.split(',')[0]
        else:
            # 没有设置的话限制在单卡上，防止多进程时占用别的卡
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ[USER_CUDA_VISIBLE_DEVICES] = ",".join(map(str, list(
                range(get_gpu_count())
            )))

    elif backend == 'jittor':
        assert _module_available(backend), f"You must have {backend} available to use {backend} backend."
        if "log_silent" not in os.environ:
            os.environ["log_silent"] = "1"
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ["use_cuda"] = "1"

    elif backend == 'torch':
        assert _module_available(backend), f"You must have {backend} available to use {backend} backend."


def set_env(global_seed=None):
    """
    set_env 用于显式告知 fastNLP 将要使用的相关环境变量是什么，必须在代码最开端运行。以下的环境变量设置，优先级分别为：（1）在代码开始
        的位置显式调用设置；（2）通过环境变量注入的；（3）通过读取配置文件（如果有）。

    :param backend: 目前支持的 backend 有 torch, jittor, paddle 。设置特定的 backend 后，fastNLP 将不再加载其它 backend ，可以
        提高加载速度。该值对应环境变量中的 FASTNLP_BACKEND 。
    :param int global_seed: 对应环境变量为 FASTNLP_GLOBAL_SEED 。设置 fastNLP 的全局随机数。
    :param str log_level: 可选 ['INFO','WARNING', 'DEBUG', 'ERROR'] ，对应环境变量为 FASTNLP_LOG_LEVEL 。
    :return:
    """

    _need_set_envs = [FASTNLP_GLOBAL_SEED]
    _env_values = defaultdict(list)

    if global_seed is not None:
        assert isinstance(global_seed, int)
        _env_values[FASTNLP_GLOBAL_SEED].append(global_seed)

    # 直接读取环境变量的，这里应当是用户自己注入的环境变量
    for env_name in _need_set_envs:
        if env_name in os.environ:
            _env_values[env_name].append(os.environ.get(env_name))

    if FASTNLP_GLOBAL_SEED in _env_values:
        os.environ[FASTNLP_GLOBAL_SEED] = _env_values[FASTNLP_GLOBAL_SEED][0]

    # 针对不同的backend，做特定的设置
    backend = os.environ.get(FASTNLP_BACKEND, '')
    if backend == 'paddle':
        assert _module_available(backend), f"You must have {backend} available to use {backend} backend."
        if os.environ.get(FASTNLP_GLOBAL_SEED, None) is not None:
            seed_paddle_global_seed(int(os.environ.get(FASTNLP_GLOBAL_SEED)))

    if backend == 'jittor':
        assert _module_available(backend), f"You must have {backend} available to use {backend} backend."
        if os.environ.get(FASTNLP_GLOBAL_SEED, None) is not None:
            seed_jittor_global_seed(int(os.environ.get(FASTNLP_GLOBAL_SEED)))

    if backend == 'torch':
        assert _module_available(backend), f"You must have {backend} available to use {backend} backend."
        if os.environ.get(FASTNLP_GLOBAL_SEED, None) is not None:
            seed_torch_global_seed(int(os.environ.get(FASTNLP_GLOBAL_SEED)))


def seed_torch_global_seed(global_seed):
    # @yxg
    pass


def seed_paddle_global_seed(global_seed):
    # @xsh
    pass

def seed_jittor_global_seed(global_seed):
    # @xsh
    pass


def dump_fastnlp_backend(default:bool = False, backend=None):
    """
    将 fastNLP 的设置写入到 ~/.fastNLP/envs/ 文件夹下，
        若 default 为 True，则保存的文件为 ~/.fastNLP/envs/default.json 。
        如 default 为 False，则保存的文件为 ~/.fastNLP/envs/{CONDA_DEFAULT_ENV}.json ，当CONDA_DEFAULT_ENV这个环境变量不存在时
        ，报错。
    当 fastNLP 被 import 时，会默认尝试从 ~/.fastNLP/envs/{CONDA_DEFAULT_ENV}.json 读取配置文件，如果文件不存在，则尝试从
     ~/.fastNLP/envs/default.json （如果有）读取环境变量。不过这些变量的优先级低于代码运行时的环境变量注入。

    会保存的环境变量为 FASTNLP_BACKEND 。

    :param default:
    :param backend: 保存使用的 backend 为哪个值，允许的值有 ['torch', 'paddle', 'jittor']。如果为 None ，则使用环境变量中的值。
    :return:
    """
    if int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
        if default:
            env_path = os.path.join(os.path.expanduser('~'), '.fastNLP', 'envs', 'default.json')
        elif 'CONDA_DEFAULT_ENV' in os.environ:
            env_path = os.path.join(os.path.expanduser('~'), '.fastNLP', 'envs',
                                    os.environ.get('CONDA_DEFAULT_ENV') + '.json')
        else:
            raise RuntimeError("Did not found `CONDA_DEFAULT_ENV` in your environment variable.")

        os.makedirs(os.path.dirname(env_path), exist_ok=True)

        envs = {}
        assert backend in SUPPORT_BACKENDS, f"fastNLP only supports {SUPPORT_BACKENDS} right now."
        if backend is None:
            if FASTNLP_BACKEND in os.environ:
                envs[FASTNLP_BACKEND] = os.environ[FASTNLP_BACKEND]
        else:
            envs[FASTNLP_BACKEND] = backend
        if len(envs):
            with open(env_path, 'w', encoding='utf8') as f:
                json.dump(fp=f, obj=envs)

            print(f"Writing the default fastNLP backend:{envs[FASTNLP_BACKEND]} to {env_path}.")
        else:
            raise RuntimeError("No backend specified.")