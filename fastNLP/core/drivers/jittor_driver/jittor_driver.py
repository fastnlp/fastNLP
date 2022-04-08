import os
import warnings
from typing import Optional, Callable, Dict

from .utils import _build_fp16_env
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
from fastNLP.core.drivers.driver import Driver
from fastNLP.core.dataloaders import JittorDataLoader
from fastNLP.core.log import logger
from fastNLP.core.utils import apply_to_collection

if _NEED_IMPORT_JITTOR:
    import jittor as jt
    from jittor import Module
    from jittor.optim import Optimizer

    _reduces = {
        'max': jt.max,
        'min': jt.min,
        'mean': jt.mean,
        'sum': jt.sum
    }


class JittorDriver(Driver):
    r"""
    Jittor 框架的 Driver
    """

    def __init__(self, model, fp16: bool = False, **kwargs):
        if not isinstance(model, Module):
            raise ValueError(f"Parameter `model` can not be `{type(model)}` in `JittorDriver`, it should be exactly "
                             f"`jittor.Module` type.")
        super(JittorDriver, self).__init__(model)

        self.model = model

        self.auto_cast, _grad_scaler = _build_fp16_env(dummy=not fp16)
        self.grad_scaler = _grad_scaler()

    @staticmethod
    def _check_dataloader_legality(dataloader, dataloader_name, is_train: bool = False):
        # 在fastnlp中实现了JittorDataLoader
        # TODO: 是否允许传入Dataset？
        if is_train:
            if not isinstance(dataloader, JittorDataLoader):
                raise ValueError(f"Parameter `{dataloader_name}` should be 'JittorDataLoader' type, not {type(dataloader)}.")
        else:
            if not isinstance(dataloader, Dict):
                raise ValueError(f"Parameter `{dataloader_name}` should be 'Dict' type, not {type(dataloader)}.")
            else:
                for each_dataloader in dataloader.values():
                    if not isinstance(each_dataloader, JittorDataLoader):
                        raise ValueError(f"Each dataloader of parameter `{dataloader_name}` should be 'JittorDataLoader' "
                                         f"type, not {type(each_dataloader)}.")

    @staticmethod
    def _check_optimizer_legality(optimizers):
        for each_optimizer in optimizers:
            if not isinstance(each_optimizer, Optimizer):
                raise ValueError(f"Each optimizer of parameter `optimizers` should be 'jittor.optim.Optimizer' type, "
                                 f"not {type(each_optimizer)}.")

    def check_evaluator_mode(self, mode: str):
        model = self.unwrap_model()
        if mode == "validate":
            if not hasattr(model, "validate_step"):
                if hasattr(model, "test_step"):
                    logger.warning(
                        "Your model does not have 'validate_step' method but has 'test_step' method, but you"
                        "are using 'mode=validate', we are going to use 'test_step' to substitute for"
                        "'validate_step'.")

        else:
            if not hasattr(model, "test_step"):
                if hasattr(model, "validate_step"):
                    logger.warning("Your model does not have 'test_step' method but has 'validate' method, but you"
                                   "are using 'mode=test', we are going to use 'validate_step' to substitute for"
                                   "'test_step'.")

    def save_model(self, filepath: str, only_state_dict: bool = False, model_save_fn: Optional[Callable]=None):
        """
        保存模型
        """
        if model_save_fn is not None:
            outputs = model_save_fn(filepath)
            if outputs is not None:
                jt.save(outputs, filepath)
        else:
            if only_state_dict:
                states = self.model.state_dict()
            else:
                warnings.warn("Saving the whole model is not supported now in Jittor. Save state dict instead.")
            jt.save(states, filepath)

    def load_model(self, filepath: str):
        """
        加载模型的加载函数；

        :param file_path: 保存文件的文件位置（需要包括文件名）；
        :return: 加载后的state_dict
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError("Checkpoint at {} not found.".format(filepath))
        return jt.load(filepath)

    def save(self):
        ...

    def load(self):
        ...

    def get_evaluate_context(self):
        return jt.no_grad

    def get_model_device(self):
        return self.model_device

    @staticmethod
    def tensor_to_numeric(tensor, reduce=None):
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
        jittor暂时没有提供数据迁移的函数，因此这个函数只是简单地返回batch
        """
        return batch

    # def set_sampler_epoch(self, dataloader: JittorDataLoader, cur_epoch_idx):
    #     # 保证 ddp 训练时的 shuffle=True 时的正确性，因为需要保证每一个进程上的 sampler 的shuffle 的随机数种子是一样的；
    #     if callable(getattr(dataloader.batch_sampler, "set_epoch", None)):
    #         dataloader.batch_sampler.set_epoch(cur_epoch_idx)