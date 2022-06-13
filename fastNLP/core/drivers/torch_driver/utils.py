import os

from typing import Any, Dict, Optional
from enum import IntEnum
import contextlib
import random
import numpy as np
import inspect

from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.envs.utils import get_global_seed
from fastNLP.envs import (
    get_global_rank,
    FASTNLP_BACKEND_LAUNCH,
    FASTNLP_GLOBAL_SEED,
)
from fastNLP.core.samplers import re_instantiate_sampler, ReproducibleBatchSampler
from fastNLP.core.utils import auto_param_call
from fastNLP.core.log import logger

if _NEED_IMPORT_TORCH:
    import torch
    # import torch.nn as nn
    from torch.nn import Module
    from torch.utils.data import DataLoader
    from torch.utils.data import RandomSampler as TorchRandomSampler
    from torch.utils.data import SequentialSampler as TorchSequentialSampler
    from torch.utils.data import BatchSampler as TorchBatchSampler

else:
    from fastNLP.core.utils.dummy_class import DummyClass as Module


__all__ = [
    'torch_seed_everything',
    'optimizer_state_to_device'
]

def torch_seed_everything(seed: int = None, add_global_rank_to_seed: bool = True) -> int:
    r"""
    为 **torch**、**numpy**、**python.random** 伪随机数生成器设置种子。

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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


class ForwardState(IntEnum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2
    PREDICT = 3


class _DDPWrappingModel(Module):
    """
    该函数用于 DDP 训练时处理用户自己定制的 train_step 等函数；
    之所以要使用这一额外的包裹模型，是因为在使用 DDP 时，必须使用 DistributedDataParallel 的 forward 函数才能实现正常的运行；
    另一方面，我们要求用户在使用我们的框架时，需要针对不用的模式实现不同的处理函数，例如 'train_step', 'evaluate_step' 等；
    然而，当使用 DistributedDataParallel 包裹 model 后，模型看不见其除了 forward 之外的方法；并且当我们尝试在训练过程中主动提取
    `model = model.module`，这同样会导致错误，会使得每一个gpu上的模型参数不同；

    因此出于以上考虑，我们实现了这一函数；
    对于更详细的解释，可以参考 'pytorch_lightning' 的 ddp 的设计；
    """

    def __init__(self, model: Module):
        super(_DDPWrappingModel, self).__init__()
        self.model = model

    def forward(self, batch, **kwargs) -> Dict:
        """
        pytorch lightning 实现了先 unwrapping_model 的操作，但是感觉对于我们来说没有什么必须要，先写个注释放这里，之后有需求了再看；
        """
        fn = kwargs.pop("fastnlp_fn")
        signature_fn = kwargs.pop("fastnlp_signature_fn")
        wo_auto_param_call = kwargs.pop("wo_auto_param_call")

        if isinstance(batch, Dict) and not wo_auto_param_call:
            return auto_param_call(fn, batch, signature_fn=signature_fn)
        else:
            return fn(batch)


class DummyGradScaler:
    """
    用于Dummy pytorch的GradScaler对象，防止重复写大量的if判断

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
        autocast = contextlib.ExitStack
        GradScaler = DummyGradScaler
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Pytorch is not installed in gpu version, please use device='cpu'.")
        if torch.cuda.get_device_capability(0)[0] < 7:
            logger.rank_zero_warning(
                "NOTE: your device does NOT support faster training with fp16, "
                "please switch to FP32 which is likely to be faster"
            )
        try:
            from torch.cuda.amp import autocast, GradScaler
        except ImportError:
            raise RuntimeError("torch version too low (less than 1.6)")
    return autocast, GradScaler


def replace_sampler(dataloader: "DataLoader", sampler):
    r"""
    替换 sampler （初始化一个新的 dataloader 的逻辑在于）：

    用户可能继承了 dataloader，定制了自己的 dataloader 类，这也是我们为什么先 `inspect.signature(dataloader)` 而不是直接
    `inspect.signature(DataLoader)` 的原因，因此同时注意到我们在外层重新初始化一个 dataloader 时也是使用的用户传进来的 dataloader
    的类，而不是直接的 DataLoader；

    如果需要定制自己的 dataloader，保证以下两点：

        1. 在 __init__ 方法中加入 **kwargs，这是为了方便我们将 sampler 插入到具体的 DataLoader 的构造中；
        2. 在 __init__ 方法中出现的参数，请务必挂为同样名字的实例属性，例如 self.one_arg_name = one_arg_name，这是因为我们只能通过属性
        来获取实际的参数的值；

     """

    # 拿到实例属性；
    instance_attrs = {k: v for k, v in vars(dataloader).items() if not k.startswith('_')}

    # 'multiprocessing_context' 是 user-defined function;
    instance_attrs["multiprocessing_context"] = dataloader.multiprocessing_context

    # 拿到 dataloader '__init__' 函数的默认函数签名；
    init_params = dict(inspect.signature(dataloader.__init__).parameters)

    # 这里为什么要单独弄的原因在于，用户在定制自己的 dataloader 的同时可能为了方便只设定一些参数，而后面直接使用 **kwargs 的方式，这时如果
    # 其在初始化自己的 dataloader 实例的时候加入了一些其它的新的参数（首先这一步是必要的，因为我们只能通过这样加 sampler；另一方面，用户
    # 可能确实通过 **kwargs 加入了一些新的参数），如果假设用户是这样使用的： "super().__init__(**kwargs)"，那么我们就只能去 DataLoader
    # 中寻找；
    has_variadic_kwargs = any(v.kind is v.VAR_KEYWORD for k, v in init_params.items())
    if has_variadic_kwargs:
        # 这里之所以这样写是因为用户自己定制的 Dataloader 中名字一样的参数所设置的默认值可能不同；因此不能直接使用 update 覆盖掉了；
        for key, value in dict(inspect.signature(DataLoader.__init__).parameters).items():
            if key not in init_params and key != 'self':
                init_params[key] = value

    # 如果初始化dataloader所使用的参数不是默认值，那么我们需要将其记录下来用于重新初始化时设置；
    non_default_params = {name for name, p in init_params.items() if
                          name in instance_attrs and p.default != instance_attrs[name]}
    # add `dataset` as it might have been replaced with `*args`
    non_default_params.add("dataset")

    reconstruct_args = {k: v for k, v in instance_attrs.items() if k in non_default_params}
    reconstruct_args.update({"sampler": sampler, "shuffle": False, "batch_sampler": None})

    batch_sampler = getattr(dataloader, "batch_sampler")
    if batch_sampler is not None and isinstance(batch_sampler, ReproducibleBatchSampler):
        raise RuntimeError("It should not be running here, please report a bug to us.")

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
            f"Trying to inject `DistributedSampler` into the `{dataloader_self_name}` instance. "
            "This would fail as some of the `__init__` arguments are not available as instance attributes. "
            f"The missing attributes are {required_args}. "
            f"HINT: If you wrote the `{dataloader_self_name}` class, define `self.missing_arg_name` or "
            "manually add the `DistributedSampler` as: "
            f"`{dataloader_self_name}(dataset, sampler=DistributedSampler(dataset))`."
        )

    # 这种错误针对的是传入的 dataloader 不是直接的 DataLoader，而是定制了 DataLoader，但是 __init__ 中没有 **kwargs；
    if not has_variadic_kwargs:

        # the dataloader signature does not allow keyword arguments that need to be passed
        missing_kwargs = reconstruct_args.keys() - init_params.keys()
        if missing_kwargs:
            missing_kwargs = sorted(missing_kwargs)
            dataloader_self_name = dataloader.__class__.__name__
            raise Exception(
                f"Trying to inject `DistributedSampler` into the `{dataloader_self_name}` instance. "
                "This would fail as it doesn't expose all its attributes in the `__init__` signature. "
                f"The missing arguments are {missing_kwargs}. "
                f"HINT: If you wrote the `{dataloader_self_name}` class, add the `__init__` arguments or "
                "manually add the `DistributedSampler` as: "
                f"`{dataloader_self_name}(dataset, sampler=DistributedSampler(dataset))`."
            )
    return type(dataloader)(**reconstruct_args)


def replace_batch_sampler(dataloader, new_batch_sampler):
    r"""
    替换一个 dataloader 的 batch_sampler；
    """
    params_keys = [k for k in dataloader.__dict__.keys() if not k.startswith("_")]
    for k in ["batch_size", "sampler", "drop_last", "batch_sampler", "dataset_kind"]:
        if k in params_keys:
            params_keys.remove(k)
    params = {k: getattr(dataloader, k) for k in params_keys}
    params["batch_sampler"] = new_batch_sampler
    return type(dataloader)(**params)


def optimizer_state_to_device(state, device):
    r"""
    将一个 ``optimizer`` 的 ``state_dict`` 迁移到对应的设备；

    :param state: ``optimzier.state_dict()``；
    :param device: 要迁移到的目的设备；
    :return: 返回迁移后的新的 state_dict；
    """
    new_state = {}
    for name, param in state.items():
        if isinstance(param, dict):
            new_state[name] = optimizer_state_to_device(param, device)
        elif isinstance(param, torch.Tensor):
            new_state[name] = param.to(device).clone()
        else:
            new_state[name] = param
    return new_state


def _check_dataloader_args_for_distributed(args, controller='Trainer'):
    if type(args.batch_sampler) is not TorchBatchSampler or (type(args.sampler) not in {TorchRandomSampler,
                                                              TorchSequentialSampler}):
        mode = 'training' if controller == 'Trainer' else 'evaluation'
        substitution = 'fastNLP.RandomSampler' if controller == 'Trainer' else 'fastNLP.UnrepeatedSequentialSampler'
        raise TypeError(f"Using customized ``batch_sampler`` or ``sampler`` for distributed {mode} may cause "
                        f"unpredictable problems, because fastNLP will substitute the dataloader's sampler into "
                        f"``{substitution}``. The customized sampler should set for distributed running  "
                        f"before initializing ``{controller}`` , and then set the "
                        f"parameter ``use_dist_sampler`` of ``{controller}`` to ``False``.")
