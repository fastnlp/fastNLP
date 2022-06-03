import inspect
import os
import random
from copy import deepcopy
from typing import Union

import numpy as np

from fastNLP.core.dataloaders import JittorDataLoader
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
from fastNLP.envs.utils import get_global_seed
from fastNLP.envs import (
    get_global_rank,
    FASTNLP_BACKEND_LAUNCH,
    FASTNLP_GLOBAL_SEED,
)
from fastNLP.core.log import logger

if _NEED_IMPORT_JITTOR:
    import jittor as jt
    from jittor.dataset import Dataset

__all__ = [
    "jittor_seed_everything",
]

def jittor_seed_everything(seed: int = None, add_global_rank_to_seed: bool = True) -> int:
    r"""
    为 **jittor**、**numpy**、**python.random** 伪随机数生成器设置种子。

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
    jt.set_global_seed(seed)
    return seed

def replace_batch_sampler(dataloader, batch_sampler):
    raise NotImplementedError("Jittor does not support using batch_sampler in `Dataset` now, "
                            "please check if you have set `Dataset.sampler` as `BatchSampler`"
                            "or report this bug to us.")

def replace_sampler(dataloader: Union["Dataset", "JittorDataLoader"], sampler):
    if isinstance(dataloader, JittorDataLoader):
        init_params = dict(inspect.signature(dataloader.__init__).parameters)
        reconstruct_args = {name: getattr(dataloader, name, p.default) for name, p in init_params.items()}
        reconstruct_args["dataset"] = replace_sampler(reconstruct_args["dataset"].dataset, reconstruct_args["dataset"].sampler)
        new_dataloader = type(dataloader)(**reconstruct_args)
        new_dataloader.dataset.set_attrs(sampler=sampler)
    else:
        new_dataloader = deepcopy(dataloader)
        new_dataloader.set_attrs(sampler=sampler)

    return new_dataloader