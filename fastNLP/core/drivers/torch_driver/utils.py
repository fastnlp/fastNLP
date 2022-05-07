import os

from typing import Any, Dict, Optional
from enum import IntEnum
import contextlib
import random
import numpy as np
import inspect

from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.core.samplers import re_instantiate_sampler

if _NEED_IMPORT_TORCH:
    import torch
    # import torch.nn as nn
    from torch.nn import Module
    from torch.utils.data import DataLoader, BatchSampler
    from torch.utils.data.sampler import Sampler
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Module


__all__ = [
    'torch_seed_everything',
    'optimizer_state_to_device'
]

from fastNLP.core.utils import auto_param_call
from fastNLP.envs import FASTNLP_GLOBAL_SEED, FASTNLP_SEED_WORKERS
from fastNLP.core.log import logger


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)


def torch_seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    r"""
    为伪随机数生成器设置种子的函数：pytorch、numpy、python.random 另外，
    设置以下环境变量：

    :param seed: 全局随机状态的整数值种子。如果为“无”，将从 "FASTNLP_GLOBAL_SEED" 环境变量中读取种子或随机选择。
    :param workers: 如果设置为“True”，将正确配置所有传递给带有“worker_init_fn”的培训师。如果用户已经提供了这样的功能对于他们的数据加载器，
     设置此参数将没有影响;
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        env_seed = os.environ.get(FASTNLP_GLOBAL_SEED)
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                # rank_zero_warn(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        logger.rank_zero_warning("Your seed value is two big or two small for numpy, we will choose a random seed for you.")

        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ[FASTNLP_SEED_WORKERS] = f"{int(workers)}"
    return seed


def reset_seed() -> None:
    r"""
    这个函数主要是给 ddp 用的，因为 ddp 会开启多个进程，因此当用户在脚本中指定 seed_everything 时，在开启多个脚本后，会在每个脚本内重新
    进行随机数的设置；
    """
    seed = os.environ.get(FASTNLP_GLOBAL_SEED, None)
    workers = os.environ.get(FASTNLP_SEED_WORKERS, "0")
    if seed is not None:
        torch_seed_everything(int(seed), workers=bool(int(workers)))


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
            raise RuntimeError("No cuda")
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
    """
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
        init_params.update(dict(inspect.signature(DataLoader.__init__).parameters))
        del init_params["self"]

    # 因为我们刚才可能用 DataLoader 的默认参数将用户定制的 dataloader 的参数覆盖掉了，因此需要重新弄一遍；
    non_default_params = {name for name, p in init_params.items() if
                          name in instance_attrs and p.default != instance_attrs[name]}
    # add `dataset` as it might have been replaced with `*args`
    non_default_params.add("dataset")

    reconstruct_args = {k: v for k, v in instance_attrs.items() if k in non_default_params}
    reconstruct_args.update(_dataloader_init_kwargs_resolve_sampler(dataloader, sampler))

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


def _dataloader_init_kwargs_resolve_sampler(
        dataloader: "DataLoader", sampler: Optional["Sampler"]
) -> Dict[str, Any]:
    """
    此函数用于处理与 DataLoader 关联的采样器、batch_sampler 参数重新实例化；
    """
    batch_sampler = getattr(dataloader, "batch_sampler")
    # checking the batch sampler type is different than PyTorch default.
    if batch_sampler is not None and not isinstance(batch_sampler, BatchSampler):
        batch_sampler = re_instantiate_sampler(batch_sampler)

        return {
            "sampler": None,
            "shuffle": False,
            "batch_sampler": batch_sampler,
            "batch_size": 1,
            "drop_last": False,
        }

    return {"sampler": sampler, "shuffle": False, "batch_sampler": None}


def replace_batch_sampler(dataloader, new_batch_sampler):
    """Helper function to replace current batch sampler of the dataloader by a new batch sampler. Function returns new
    dataloader with new batch sampler.

    Args:
        dataloader: input dataloader
        new_batch_sampler: new batch sampler to use

    Returns:
        DataLoader
    """
    params_keys = [k for k in dataloader.__dict__.keys() if not k.startswith("_")]
    for k in ["batch_size", "sampler", "drop_last", "batch_sampler", "dataset_kind"]:
        if k in params_keys:
            params_keys.remove(k)
    params = {k: getattr(dataloader, k) for k in params_keys}
    params["batch_sampler"] = new_batch_sampler
    return type(dataloader)(**params)
    # TODO 这里是否可以auto_param_call一下
    # return auto_param_call(type(dataloader), params, {'self': type(dataloader).__new__()},
    #                        signature_fn=type(dataloader).__init__)


def optimizer_state_to_device(state, device):
    new_state = {}
    for name, param in state.items():
        if isinstance(param, dict):
            new_state[name] = optimizer_state_to_device(param, device)
        elif isinstance(param, torch.Tensor):
            new_state[name] = param.to(device).clone()
        else:
            new_state[name] = param
    return new_state




