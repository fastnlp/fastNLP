import socket
import os
import struct
import random
import inspect
import numpy as np
from contextlib import ExitStack, closing
from enum import IntEnum
from typing import Dict, Optional, Union

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.core.utils import get_paddle_device_id, auto_param_call, paddle_to
from fastNLP.envs.env import FASTNLP_GLOBAL_SEED, FASTNLP_SEED_WORKERS, USER_CUDA_VISIBLE_DEVICES
from fastNLP.core.log import logger


if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle import nn
    from paddle.nn import Layer
    from paddle.io import DataLoader, BatchSampler
    from paddle.amp import auto_cast, GradScaler
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Layer


__all__ = [
    "paddle_seed_everything",
]

def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)

def paddle_seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:

    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        env_seed = os.environ.get("GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            # rank_zero_warn(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                # rank_zero_warn(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        logger.warning("Your seed value is two big or two small for numpy, we will choose a random seed for "
                        "you.")

        # rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    # log.info(f"Global seed set to {seed}")
    os.environ[FASTNLP_GLOBAL_SEED] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # paddle的seed函数会自行判断是否在gpu环境，如果在的话会设置gpu的种子
    paddle.seed(seed)
    os.environ[FASTNLP_SEED_WORKERS] = f"{int(workers)}"
    return seed


def reset_seed() -> None:
    """
    fleet 会开启多个进程，因此当用户在脚本中指定 seed_everything 时，在开启多个脚本后，会在每个脚本内重新
    进行随机数的设置；
    """
    seed = os.environ.get(FASTNLP_GLOBAL_SEED, None)
    workers = os.environ.get(FASTNLP_SEED_WORKERS, "0")
    if seed is not None:
        paddle_seed_everything(int(seed), workers=bool(int(workers)))

class ForwardState(IntEnum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2
    PREDICT = 3

_MODE_PARAMETER = "_forward_state"

class _FleetWrappingModel(Layer):
    """
    参考_DDPWrappingModel，paddle的分布式训练也需要用paddle.nn.DataParallel进行包装，采用和
    pytorch相似的处理方式
    """
    def __init__(self, model: 'nn.Layer'):
        super(_FleetWrappingModel, self).__init__()
        self.model = model

        if isinstance(model, paddle.DataParallel):
            model = model._layers
            if hasattr(model, "train_step"):
                logger.warning(
                    "Notice your model is a `paddle.DataParallel` model. And your "
                    "model also implements the `train_step` method, which we can not call actually, we will"
                    " call `forward` function instead of `train_step` and you should note that.")
            self._train_step = self.model
            self._train_signature_fn = model.forward

            if hasattr(model, "validate_step"):
                logger.warning(
                    "Notice your model is a `paddle.DataParallel` model. And your "
                    "model also implements the `validate_step` method, which we can not call actually, "
                    "we will call `forward` function instead of `validate_step` and you should note that.")
            self._validate_step = self.model
            self._validate_signature_fn = model.forward

            if hasattr(model, "test_step"):
                logger.warning(
                    "Notice your model is a `paddle.DataParallel` model. And your "
                    "model also implements the `test_step` method, which we can not call actually, we will"
                    " call `forward` function instead of `test_step` and you should note that.")
            self._test_step = self.model
            self._test_signature_fn = model.forward
        else:
            if hasattr(model, "train_step"):
                self._train_step = model.train_step
                self._train_signature_fn = None
            else:
                self._train_step = model
                self._train_signature_fn = model.forward

            if hasattr(model, "validate_step"):
                self._validate_step = model.validate_step
                self._validate_signature_fn = None
            elif hasattr(model, "test_step"):
                self._validate_step = model.test_step
                self._validate_signature_fn = None
            else:
                self._validate_step = model
                self._validate_signature_fn = model.forward

            if hasattr(model, "test_step"):
                self._test_step = model.test_step
                self._test_signature_fn = None
            elif hasattr(model, "validate_step"):
                self._test_step = model.validate_step
                self._test_signature_fn = None
            else:
                self._test_step = model
                self._test_signature_fn = model.forward

    def forward(self, batch, **kwargs) -> Dict:

        _forward_state = kwargs.pop(_MODE_PARAMETER)

        if _forward_state == ForwardState.TRAIN:
            if isinstance(batch, Dict):
                return auto_param_call(self._train_step, batch, signature_fn=self._train_signature_fn)
            else:
                return self._train_step(batch)
        elif _forward_state == ForwardState.VALIDATE:
            if isinstance(batch, Dict):
                return auto_param_call(self._validate_step, batch, signature_fn=self._validate_signature_fn)
            else:
                return self._validate_step(batch)
        elif _forward_state == ForwardState.TEST:
            if isinstance(batch, Dict):
                return auto_param_call(self._test_step, batch, signature_fn=self._test_signature_fn)
            else:
                return self._test_step(batch)
        elif _forward_state == ForwardState.PREDICT:
            raise NotImplementedError("'PREDICT' mode has not been implemented.")
        else:
            raise NotImplementedError("You should direct a concrete mode.")

class DummyGradScaler:
    """
    用于仿造的GradScaler对象，防止重复写大量的if判断

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
        auto_cast = ExitStack
        GradScaler = DummyGradScaler
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

def get_host_name_ip():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return host_name, host_ip
    except:
        return None

def get_device_from_visible(device: Union[str, int], output_type=int):
    """
    在有 CUDA_VISIBLE_DEVICES 的情况下，获取对应的设备。
    如 CUDA_VISIBLE_DEVICES=2,3 ，device=3 ，则返回1。
    :param devices: 未转化的设备名
    :param output_type: 返回值的类型
    :return: 转化后的设备id
    """
    if output_type not in [int, str]:
        raise ValueError("Parameter `output_type` should be one of these types: [int, str]")
    if device == "cpu":
        return device
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    idx = get_paddle_device_id(device)
    if cuda_visible_devices is None or cuda_visible_devices == "":
        # 这个判断一般不会发生，因为 fastnlp 会为 paddle 强行注入 CUDA_VISIBLE_DEVICES
        raise RuntimeError("This situation should not happen, please report us this bug.")
    else:
        # 利用 USER_CUDA_VISIBLDE_DEVICES 获取用户期望的设备
        user_visible_devices = os.getenv(USER_CUDA_VISIBLE_DEVICES)
        if user_visible_devices is None:
            raise RuntimeError("This situation cannot happen, please report a bug to us.")
        idx = user_visible_devices.split(",")[idx]

        cuda_visible_devices_list = cuda_visible_devices.split(',')
        if idx not in cuda_visible_devices_list:
            raise ValueError(f"Can't find your devices {idx} in CUDA_VISIBLE_DEVICES[{cuda_visible_devices}].")
        res = cuda_visible_devices_list.index(idx)
        if output_type == int:
            return res
        else:
            return f"gpu:{res}"

def replace_batch_sampler(dataloader: "DataLoader", batch_sampler: "BatchSampler"):
    """
    利用 `batch_sampler` 重新构建一个 DataLoader，起到替换 `batch_sampler` 又不影响原 `dataloader` 的作用。
    考虑了用户自己定制了 DataLoader 的情形。
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
        init_params.update(dict(inspect.signature(DataLoader.__init__).parameters))
        del init_params["self"]

    # 因为我们刚才可能用 DataLoader 的默认参数将用户定制的 dataloader 的参数覆盖掉了，因此需要重新弄一遍；
    # 将同时在实例名和参数名中出现且不是默认值的参数收集起来
    non_default_params = {name for name, p in init_params.items() if
                          name in instance_attrs and p.default != instance_attrs[name]}
    # add `dataset` as it might have been replaced with `*args`
    non_default_params.add("dataset")

    # 收集不是默认值的参数和它的值
    reconstruct_args = {k: v for k, v in instance_attrs.items() if k in non_default_params}
    # persistent_workers 在类中的对应成员带有下划线，因此添加进来
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
    使用 `new_sampler` 重新构建一个 BatchSampler，并替换到 `dataloader` 中
    """
    new_batch_sampler = BatchSampler(
        dataset=dataloader.batch_sampler.dataset,
        sampler=new_sampler,
        shuffle=isinstance(dataloader.batch_sampler.sampler, paddle.io.RandomSampler),
        batch_size=dataloader.batch_sampler.batch_size,
        drop_last=dataloader.batch_sampler.drop_last
    )
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
