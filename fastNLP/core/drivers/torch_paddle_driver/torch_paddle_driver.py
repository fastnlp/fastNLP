from typing import Optional, Dict, Union, Callable

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE, _NEED_IMPORT_TORCH


if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle.io import DataLoader as PaddleDataLoader
    from paddle.optimizer import Optimizer as PaddleOptimizer

if _NEED_IMPORT_TORCH:
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.optim import Optimizer as TorchOptimizer

from fastNLP.core.drivers.driver import Driver
from fastNLP.envs.distributed import rank_zero_call
from fastNLP.core.utils.utils import auto_param_call, apply_to_collection
from fastNLP.core.log.logger import logger
from fastNLP.modules.mix_modules.mix_module import MixModule


__all__ = [
    "TorchPaddleDriver",
]

class TorchPaddleDriver(Driver):
    """
    针对torch和paddle混合模型的driver
    由于是两种不同的框架不方便实现多卡，暂时先实现CPU和GPU单卡的功能
    """
    def __init__(self, model, device: Optional[str] = None, **kwargs):
        super(TorchPaddleDriver, self).__init__(model)

        self.model_device = device
        self.torch_non_blocking = kwargs.get("torch_non_blocking", None)
        self.paddle_blocking = kwargs.get("paddle_blocking", None)

        self._data_device = kwargs.get("_data_device", None)
        if isinstance(self._data_device, int):
            # 将data_device设置为cuda:x的字符串形式
            if self._data_device < 0:
                raise ValueError("Parameter `_data_device` can not be smaller than 0.")
            _could_use_device_num = paddle.device.cuda.device_count()
            if self._data_device >= _could_use_device_num:
                raise ValueError("The gpu device that parameter `device` specifies is not existed.")
            self._data_device = f"cuda:{self._data_device}"
        elif self._data_device is not None:
            raise ValueError("Parameter `device` is wrong type, please check our documentation for the right use.")

        if hasattr(self.model, "train_step"):
            self._train_step = self.model.train_step
            self._train_signature_fn = None
        else:
            self._train_step = self.model
            self._train_signature_fn = self.model.forward

        if hasattr(self.model, "validate_step"):
            self._validate_step = self.model.validate_step
            self._validate_signature_fn = None
        elif hasattr(self.model, "test_step"):
            self._validate_step = self.model.test_step
            self._validate_signature_fn = self.model.forward
        else:
            self._validate_step = self.model
            self._validate_signature_fn = self.model.forward

        if hasattr(self.model, "test_step"):
            self._test_step = self.model.test_step
            self._test_signature_fn = None
        elif hasattr(self.model, "validate_step"):
            self._test_step = self.model.validate_step
            self._test_signature_fn = self.model.forward
        else:
            self._test_step = self.model
            self._test_signature_fn = self.model.forward

    def setup(self):
        if self.model_device is not None:
            paddle.device.set_device(self.model_device.replace("cuda", "gpu"))
            self.model.to(self.model_device)

    @staticmethod
    def _check_dataloader_legality(dataloader, dataloader_name, is_train: bool = False):
        if is_train:
            if not isinstance(dataloader, (TorchDataLoader, PaddleDataLoader)):
                raise ValueError(f"Parameter `{dataloader_name}` should be 'torch.util.data.DataLoader' or `paddle.io.dataloader` type, not {type(dataloader)}.")
        else:
            if not isinstance(dataloader, Dict):
                raise ValueError(f"Parameter `{dataloader_name}` should be 'Dict' type, not {type(dataloader)}.")
            else:
                for each_dataloader in dataloader.values():
                    if not isinstance(each_dataloader, (TorchDataLoader, PaddleDataLoader)):
                        raise ValueError(f"Each dataloader of parameter `{dataloader_name}` should be "
                                         f"'torch.util.data.DataLoader' or `paddle.io.dataloader` "
                                         f"type, not {type(each_dataloader)}.")

    @staticmethod
    def _check_optimizer_legality(optimizers):
        for each_optimizer in optimizers:
            if not isinstance(each_optimizer, (TorchOptimizer, PaddleOptimizer)):
                raise ValueError(f"Each optimizers of parameter `optimizers` should be "
                                 f"'torch.optim.Optimizer' or 'paddle.optimizers.Optimizer' type, "
                                 f"not {type(each_optimizer)}.")

    def train_step(self, batch) -> Dict:
        if isinstance(batch, Dict):
            return auto_param_call(self._train_step, batch)
        else:
            return self._train_step(batch)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def backward(self, loss):
        loss.backward()

    def zero_grad(self):
        for optimizer in self.optimizers:
            if isinstance(optimizer, TorchOptimizer):
                optimizer.zero_grad()
            elif isinstance(optimizer, PaddleOptimizer):
                optimizer.clear_grad()
            else:
                raise ValueError("Unknown optimizers type.")

    def validate_step(self, batch):
        if isinstance(batch, Dict):
            return auto_param_call(self._validate_step, batch)
        else:
            return self._validate_step(batch)

    def test_step(self, batch):
        if isinstance(batch, Dict):
            return auto_param_call(self._test_step, batch)
        else:
            return self._test_step(batch)

    def predict_step(self, batch):
        if isinstance(batch, Dict):
            return auto_param_call(self._predict_step, batch)
        else:
            return self._predict_step(batch)

    @rank_zero_call
    def save_model(self, filepath: str, only_state_dict: bool = True, model_save_fn: Optional[Callable] = None):
        r"""
        暂时不提供保存整个模型的方法
        """
        if only_state_dict == False:
            logger.warn("TorchPaddleModule only support saving state dicts now.")
        if model_save_fn is not None:
            model_save_fn(filepath)
        else:
            model = self.unwrap_model()
            self.move_model_to_device(model, "cpu")
            self.model.save(filepath)
            self.move_model_to_device(model, self.model_device)

    def load_model(self, filepath: str):
        """
        加载模型的加载函数；

        :param filepath: 保存文件的文件位置（需要包括文件名）；
        :return:
        """
        return self.model.load(filepath)

    def save(self):
        ...

    def load(self):
        ...

    @staticmethod
    def move_model_to_device(model: MixModule, device: str):
        if device is not None:
            model.to(device)

    def unwrap_model(self):
        return self.model

    @staticmethod
    def tensor_to_numeric(tensor):
        if tensor is None:
            return None

        def _translate(_data):
            return _data.tolist()

        return apply_to_collection(
            data=tensor,
            dtype=(paddle.Tensor, torch.Tensor),
            function=_translate
        )

    def set_model_mode(self, mode: str):
        assert mode in {"train", "eval"}
        getattr(self.model, mode)()

    def get_model_device(self):
        return self.model_device

    @property
    def data_device(self):
        if self.model_device is not None:
            return self.model_device
        else:
            return self._data_device

    def set_model_mode(self, mode: str):
        assert mode in {"train", "eval"}
        getattr(self.model, mode)()

    def set_sampler_epoch(self, dataloader: Union['TorchDataLoader', 'PaddleDataLoader'], cur_epoch_idx):
        # 保证 ddp 训练时的 shuffle=True 时的正确性，因为需要保证每一个进程上的 sampler 的shuffle 的随机数种子是一样的；
        return dataloader
