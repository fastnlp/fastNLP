import inspect
from copy import deepcopy
from typing import Union

from fastNLP.core.dataloaders import JittorDataLoader
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    from jittor.dataset import Dataset

__all__ = []

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