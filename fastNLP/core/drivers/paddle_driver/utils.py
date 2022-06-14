import socket
import os
import struct
import random
import inspect
import numpy as np
from copy import deepcopy
from contextlib import ExitStack, closing
from typing import Dict, Optional

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.envs.utils import get_global_seed
from fastNLP.envs import (
    get_global_rank,
    FASTNLP_BACKEND_LAUNCH,
    FASTNLP_GLOBAL_SEED,
)
from fastNLP.core.utils import auto_param_call, paddle_to
from fastNLP.core.log import logger


if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle import nn
    from paddle.nn import Layer
    from paddle.io import DataLoader, BatchSampler, RandomSampler, SequenceSampler
    from paddle.amp import auto_cast, GradScaler
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Layer


__all__ = [
    "paddle_seed_everything",
]

def paddle_seed_everything(seed: int = None, add_global_rank_to_seed: bool = True) -> int:
    r"""
    为 **paddle**、**numpy**、**python.random** 伪随机数生成器设置种子。

    :param seed: 全局随机状态的整数值种子。如果为 ``None`` 则会根据时间戳生成一个种子。
    :param add_global_rank_to_seed: 在分布式训练中，是否在不同 **rank** 中使用不同的随机数。
        当设置为 ``True`` 时，**FastNLP** 会将种子加上当前的 ``global_rank``。
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        if os.getenv(FASTNLP_BACKEND_LAUNCH) == "1":
            seed = 42
        else:
            seed = get_global_seed()
        logger.info(f"'FASTNLP_GLOBAL_SEED' is set to {seed} automatically.")
    if not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        logger.rank_zero_warning("Your seed value is too big or too small for numpy, we will choose a random seed for you.")
        seed %= max_seed_value

    os.environ[FASTNLP_GLOBAL_SEED] = f"{seed}"
    if add_global_rank_to_seed:
        seed += get_global_rank()

    random.seed(seed)
    np.random.seed(seed)
    # paddle的seed函数会自行判断是否在gpu环境，如果在的话会设置gpu的种子
    paddle.seed(seed)
    return seed

class _FleetWrappingModel(Layer):
    """
    参考 :class:`fastNLP.core.drivers.torch_driver.utils._DDPWrappingModel` ， **PaddlePaddle** 的分布式训练也需要用 :class:`paddle.nn.DataParallel` 进行包装，采用和
    **pytorch** 相似的处理方式
    """
    def __init__(self, model: 'nn.Layer'):
        super(_FleetWrappingModel, self).__init__()
        self.model = model

    def forward(self, batch, **kwargs) -> Dict:

        fn = kwargs.pop("fastnlp_fn")
        signature_fn = kwargs.pop("fastnlp_signature_fn")
        wo_auto_param_call = kwargs.pop("wo_auto_param_call")

        if isinstance(batch, Dict) and not wo_auto_param_call:
            return auto_param_call(fn, batch, signature_fn=signature_fn)
        else:
            return fn(batch)

class DummyGradScaler:
    """
    用于仿造的 **GradScaler** 对象，防止重复写大量的if判断
    """
    def __init__(self, *args, **kwargs):
        pass

    def get_scale(self):
        return 1.0

    def is_enabled(self):
        return False

    def scale(self, outputs):
        return outputs

    def step(self, optimizer, *args, **kwargs):
        optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        pass

    def unscale_(self, optimizer):
        pass

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {}

def _build_fp16_env(dummy=False):
    if dummy:
        return ExitStack, DummyGradScaler
    else:
        if not paddle.device.is_compiled_with_cuda():
            raise RuntimeError("No cuda")
        if paddle.device.cuda.get_device_capability(0)[0] < 7:
            logger.warning(
                "NOTE: your device does NOT support faster training with fp16, "
                "please switch to FP32 which is likely to be faster"
            )
            return auto_cast, GradScaler

def find_free_ports(num):
    """
    在空闲的端口中找到 ``num`` 个端口
    """
    def __free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                         struct.pack('ii', 1, 0))
            s.bind(('', 0))
            return s.getsockname()[1]

    port_set = set()
    step = 0
    while True:
        port = __free_port()
        if port not in port_set:
            port_set.add(port)

        if len(port_set) >= num:
            return port_set

        step += 1
        if step > 400:
            logger.error(
                "can't find avilable port and use the specified static port now!"
            )
            return None

    return None

def replace_batch_sampler(dataloader: "DataLoader", batch_sampler: "BatchSampler"):
    """
    利用 ``batch_sampler`` 重新构建一个 ``DataLoader``，起到替换 ``batch_sampler`` 又不影响原 ``dataloader`` 的作用。
    考虑了用户自己定制了 ``DataLoader`` 的情形。
    """
    # 拿到非下划线开头的实例属性；
    instance_attrs = {k: v for k, v in vars(dataloader).items() if not k.startswith('_')}

    # 拿到 dataloader '__init__' 函数的默认函数签名；可以获取参数名和参数的默认值以及类型
    init_params = dict(inspect.signature(dataloader.__init__).parameters)

    # 这里为什么要单独弄的原因在于，用户在定制自己的 dataloader 的同时可能为了方便只设定一些参数，而后面直接使用 **kwargs 的方式，这时如果
    # 其在初始化自己的 dataloader 实例的时候加入了一些其它的新的参数（首先这一步是必要的，因为我们只能通过这样加 sampler；另一方面，用户
    # 可能确实通过 **kwargs 加入了一些新的参数），如果假设用户是这样使用的： "super().__init__(**kwargs)"，那么我们就只能去 DataLoader
    # 中寻找；VAR_KEYWORD 代表 **kwargs
    has_variadic_kwargs = any(v.kind is v.VAR_KEYWORD for k, v in init_params.items())
    if has_variadic_kwargs:
        for key, value in dict(inspect.signature(DataLoader.__init__).parameters).items():
            if key not in init_params and key != 'self':
                init_params[key] = value

    # 如果初始化dataloader所使用的参数不是默认值，那么我们需要将其记录下来用于重新初始化时设置；
    non_default_params = {name for name, p in init_params.items() if
                          name in instance_attrs and p.default != instance_attrs[name]}
    # add `dataset` as it might have been replaced with `*args`
    non_default_params.add("dataset")

    reconstruct_args = {k: v for k, v in instance_attrs.items() if k in non_default_params}
    reconstruct_args.update({
        "batch_sampler": batch_sampler, "shuffle": False, "drop_last": False, "batch_size": 1,
        "persistent_workers": dataloader._persistent_workers,
    })

    # POSITIONAL_OR_KEYWORD 代表一般的参数
    # 收集初始化函数中出现的、一般形式的、不带默认值且不在 reconstruct_args 中的参数
    # 也即它们没有在初始化函数和实例成员中同时出现
    required_args = {
        p.name
        for p in init_params.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
           and p.default is p.empty
           and p.name not in reconstruct_args
    }

    # 这种错误针对的是 __init__ 中的参数没有用同样名字的 self 挂上；
    if required_args:
        required_args = sorted(required_args)
        dataloader_self_name = dataloader.__class__.__name__
        raise Exception(
            f"Trying to inject `BatchSampler` into the `{dataloader_self_name}` instance. "
            "This would fail as some of the `__init__` arguments are not available as instance attributes. "
            f"The missing attributes are {required_args}. "
        )

    # 这种错误针对的是传入的 dataloader 不是直接的 DataLoader，而是定制了 DataLoader，但是 __init__ 中没有 **kwargs；
    if not has_variadic_kwargs:

        # the dataloader signature does not allow keyword arguments that need to be passed
        missing_kwargs = reconstruct_args.keys() - init_params.keys()
        if missing_kwargs:
            missing_kwargs = sorted(missing_kwargs)
            dataloader_self_name = dataloader.__class__.__name__
            raise Exception(
                f"Trying to inject `BatchSampler` into the `{dataloader_self_name}` instance. "
                "This would fail as it doesn't expose all its attributes in the `__init__` signature. "
                f"The missing arguments are {missing_kwargs}. "
            )

    return type(dataloader)(**reconstruct_args)

def replace_sampler(dataloader, new_sampler):
    """
    使用 ``new_sampler`` 重新构建一个 ``BatchSampler``，并替换到 ``dataloader`` 中
    """
    new_batch_sampler = deepcopy(dataloader.batch_sampler)
    new_batch_sampler.sampler = new_sampler
    return replace_batch_sampler(dataloader, new_batch_sampler)

def optimizer_state_to_device(state, device):
    new_state = {}
    for name, param in state.items():
        if isinstance(param, dict):
            new_state[name] = optimizer_state_to_device(param, device)
        elif isinstance(param, paddle.Tensor):
            new_state[name] = paddle_to(param, device).clone()
        else:
            new_state[name] = param
    return new_state

def _check_dataloader_args_for_distributed(args, controller='Trainer'):
    if type(args.batch_sampler) is not BatchSampler or (type(args.sampler) not in {RandomSampler,
                                                              SequenceSampler}):
        mode = 'training' if controller == 'Trainer' else 'evaluation'
        substitution = 'fastNLP.RandomSampler' if controller == 'Trainer' else 'fastNLP.UnrepeatedSequentialSampler'
        raise TypeError(f"Using customized ``batch_sampler`` or ``sampler`` for distributed {mode} may cause "
                        f"unpredictable problems, because fastNLP will substitute the dataloader's sampler into "
                        f"``{substitution}``. The customized sampler should set for distributed running  "
                        f"before initializing ``{controller}`` , and then set the "
                        f"parameter ``use_dist_sampler`` of ``{controller}`` to ``False``.")
