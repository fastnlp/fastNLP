import os
import random
from pathlib import Path
from typing import Union, Optional
from functools import partial

import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
from fastNLP.core.drivers.driver import Driver
from fastNLP.core.dataloaders import JittorDataLoader
from fastNLP.core.log import logger
from fastNLP.core.utils import apply_to_collection
from fastNLP.envs import FASTNLP_GLOBAL_RANK, FASTNLP_SEED_WORKERS

if _NEED_IMPORT_JITTOR:
    import jittor as jt
    from jittor import Module
    from jittor.optim import Optimizer
    from jittor.dataset import Dataset

    _reduces = {
        'max': jt.max,
        'min': jt.min,
        'mean': jt.mean,
        'sum': jt.sum
    }

__all__ = [
    "JittorDriver",
]

class JittorDriver(Driver):
    r"""
    ``Jittor`` 框架的 ``Driver``

    .. note::

        这是一个正在开发中的功能，敬请期待。

    .. todo::

        实现 fp16 的设置，且支持 cpu 和 gpu 的切换；
        实现用于断点重训的 save 和 load 函数；

    """

    def __init__(self, model, fp16: bool = False, **kwargs):
        if not isinstance(model, Module):
            raise ValueError(f"Parameter `model` can not be `{type(model)}` in `JittorDriver`, it should be exactly "
                             f"`jittor.Module` type.")
        super(JittorDriver, self).__init__(model)

        if fp16:
            jt.flags.auto_mixed_precision_level = 6
        else:
            jt.flags.auto_mixed_precision_level = 0
        self.fp16 = fp16

        # 用来设置是否关闭 auto_param_call 中的参数匹配问题；
        self.wo_auto_param_call = kwargs.get("model_wo_auto_param_call", False)

    def check_dataloader_legality(self, dataloader):
        if not isinstance(dataloader, (Dataset, JittorDataLoader)):
            raise TypeError(f"{Dataset} or {JittorDataLoader} is expected, instead of `{type(dataloader)}`")

    @staticmethod
    def _check_optimizer_legality(optimizers):
        for each_optimizer in optimizers:
            if not isinstance(each_optimizer, Optimizer):
                raise ValueError(f"Each optimizer of parameter `optimizers` should be 'jittor.optim.Optimizer' type, "
                                 f"not {type(each_optimizer)}.")

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def backward(self, loss):
        for optimizer in self.optimizers:
            optimizer.backward(loss)

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def save_model(self, filepath: Union[str, Path], only_state_dict: bool = True, **kwargs):
        r"""
        将模型保存到 ``filepath`` 中。

        :param filepath: 保存文件的文件位置（需要包括文件名）；
        :param only_state_dict: 在 **Jittor** 中，该参数无效，**Jittor** 仅支持保存模型的 ``state_dict``。
        """
        if not only_state_dict:
            logger.rank_zero_warning(
                "Jittor only supports saving state_dict, and we will also save state_dict for you.",
                once=True
            )
        if isinstance(filepath, Path):
            filepath = str(filepath)
        model = self.unwrap_model()
        model.save(filepath)

    def load_model(self, filepath: Union[Path, str], only_state_dict: bool = True, **kwargs):
        r"""
        加载模型的函数；将 ``filepath`` 中的模型加载并赋值给当前 ``model`` 。

        :param filepath: 保存文件的文件位置（需要包括文件名）；
        :param load_state_dict: 在 **Jittor** 中，该参数无效，**Jittor** 仅支持加载模型的 ``state_dict``。
        """
        if not only_state_dict:
            logger.rank_zero_warning(
                "Jittor only supports loading state_dict, and we will also load state_dict for you.",
                once=True
            )
        if isinstance(filepath, Path):
            filepath = str(filepath)
        model = self.unwrap_model()
        model.load(filepath)

    def save_checkpoint(self):
        ...

    def get_optimizer_state(self):
        # optimizers_state_dict = {}
        # for i in range(len(self.optimizers)):
        #     optimizer: torch.optim.Optimizer = self.optimizers[i]
        #     optimizer_state = optimizer.state_dict()
        #     optimizer_state["state"] = optimizer_state_to_device(optimizer_state["state"], torch.device("cpu"))
        #     optimizers_state_dict[f"optimizer{i}"] = optimizer_state  # 注意这里没有使用 deepcopy，测试是不需要的；
        # return optimizers_state_dict
        ...

    def load_optimizer_state(self, states):
        # assert len(states) == len(self.optimizers), f"The number of optimizers is:{len(self.optimizers)}, while in " \
        #                                             f"checkpoint it is:{len(states)}"
        # for i in range(len(self.optimizers)):
        #     optimizer: torch.optim.Optimizer = self.optimizers[i]
        #     optimizer.load_state_dict(states[f"optimizer{i}"])
        # logger.debug("Load optimizer state dict.")
        ...

    def load_checkpoint(self):
        ...

    def get_evaluate_context(self):
        return jt.no_grad

    @staticmethod
    def move_model_to_device(model: "jt.Module", device):
        r"""
        将模型转移到指定的设备上。由于 **Jittor** 会自动为数据分配设备，因此该函数实际上无效。
        """
        ...

    def move_data_to_device(self, batch):
        """
        将数据 ``batch`` 转移到指定的设备上。由于 **Jittor** 会自动为数据分配设备，因此该函数实际上无效。
        """
        return batch

    @staticmethod
    def tensor_to_numeric(tensor, reduce=None):
        r"""
        将一个 :class:`jittor.Var` 对象转换为 转换成 python 中的数值类型；

        :param tensor: :class:`jittor.Var` 类型的对象；
        :param reduce: 当 tensor 是一个多数值的张量时，应当使用何种归一化操作来转换成单一数值，应当为以下类型之一：``['max', 'min', 'sum', 'mean']``；
        :return: 返回一个单一数值，其数值类型是 python 中的基本的数值类型，例如 ``int，float`` 等；
        """
        if tensor is None:
            return None

        def _translate(_data):
            # 如果只含有一个元素，则返回元素本身，而非list
            if _data.numel() == 1:
                return _data.item()
            if reduce is None:
                return _data.tolist()
            return _reduces[reduce](_data).item()

        return apply_to_collection(
            data=tensor,
            dtype=jt.Var,
            function=_translate
        )

    def set_model_mode(self, mode: str):
        assert mode in {"train", "eval"}
        getattr(self.model, mode)()

    @property
    def data_device(self):
        return self.model_device

    def move_data_to_device(self, batch: 'jt.Var'):
        """
        **jittor** 暂时没有提供数据迁移的函数，因此这个函数只是简单地返回 **batch**
        """
        return batch

    @staticmethod
    def worker_init_function(worker_id: int, rank: Optional[int] = None) -> None:  # pragma: no cover
        global_rank = rank if rank is not None else int(os.environ.get(FASTNLP_GLOBAL_RANK, 0))
        process_seed = jt.get_seed()
        # back out the base seed so we can use all the bits
        base_seed = process_seed - worker_id
        ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
        # use 128 bits (4 x 32-bit words)
        np.random.seed(ss.generate_state(4))
        # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
        jittor_ss, stdlib_ss = ss.spawn(2)
        jt.set_global_seed(jittor_ss.generate_state(1, dtype=np.uint64)[0])
        # use 128 bits expressed as an integer
        stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
        random.seed(stdlib_seed)

    def set_deterministic_dataloader(self, dataloader: Union["JittorDataLoader", "Dataset"]):
        if int(os.environ.get(FASTNLP_SEED_WORKERS, 0)) and dataloader.worker_init_fn is None:
            dataloader.worker_init_fn = partial(self.worker_init_function,
                                                rank=int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)))

    def set_sampler_epoch(self, dataloader: Union["JittorDataLoader", "Dataset"], cur_epoch_idx: int):
        # 保证 ddp 训练时的 shuffle=True 时的正确性，因为需要保证每一个进程上的 sampler 的shuffle 的随机数种子是一样的；
        if callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(cur_epoch_idx)

    @staticmethod
    def get_dataloader_args(dataloader: Union["JittorDataLoader", "Dataset"]):
        pass
