import os
from pathlib import Path
from typing import Union, Optional, Dict
from dataclasses import dataclass

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
from fastNLP.core.drivers.driver import Driver
from fastNLP.core.dataloaders import JittorDataLoader
from fastNLP.core.dataloaders import OverfitDataLoader
from fastNLP.core.samplers import ReproducibleSampler, RandomSampler
from fastNLP.core.log import logger
from fastNLP.core.utils import apply_to_collection, nullcontext
from fastNLP.envs import (
    FASTNLP_MODEL_FILENAME,
    FASTNLP_CHECKPOINT_FILENAME,
)

if _NEED_IMPORT_JITTOR:
    import jittor as jt
    from jittor import Module
    from jittor.optim import Optimizer
    from jittor.dataset import Dataset
    from jittor.dataset import (
        BatchSampler as JittorBatchSampler,
        Sampler as JittorSampler,
        RandomSampler as JittorRandomSampler,
        SequentialSampler as JittorSequentialSampler
    )

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
        self._auto_cast = nullcontext

        # 用来设置是否关闭 auto_param_call 中的参数匹配问题；
        self.wo_auto_param_call = kwargs.get("model_wo_auto_param_call", False)

    def check_dataloader_legality(self, dataloader):
        if not isinstance(dataloader, (Dataset, JittorDataLoader, OverfitDataLoader)):
            raise TypeError(f"{Dataset} or {JittorDataLoader} is expected, instead of `{type(dataloader)}`")
        if len(dataloader) == 0:
            logger.rank_zero_warning("Your dataloader is empty, which is not recommended because it "
                                        "may cause some unexpected exceptions.", once=True)

    @staticmethod
    def _check_optimizer_legality(optimizers):
        for each_optimizer in optimizers:
            if not isinstance(each_optimizer, Optimizer):
                raise TypeError(f"Each optimizer of parameter `optimizers` should be 'jittor.optim.Optimizer' type, "
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

    def save_checkpoint(self, folder: Path, states: Dict, dataloader, only_state_dict: bool = True, should_save_model: bool = True, **kwargs):
        dataloader_args = self.get_dataloader_args(dataloader)
        if dataloader_args.sampler:
            sampler = dataloader_args.sampler
        else:
            raise RuntimeError("This condition is not supposed to appear. Please report a bug to us.")
        
        num_consumed_batches = states.pop('num_consumed_batches')
        if hasattr(sampler, 'state_dict') and callable(sampler.state_dict):
            sampler_states = sampler.state_dict()
            if dataloader_args.batch_size is not None:
                sampler_states['num_consumed_samples'] = sampler.num_replicas * dataloader_args.batch_size \
                                                            * num_consumed_batches
            else:
                logger.rank_zero_warning("fastNLP cannot get batch_size, we have to save based on `num_consumed_samples`, "
                                "it may cause missing some samples when reload.")

            states['sampler_states'] = sampler_states
        else:
            raise RuntimeError('The sampler has no `state_dict()` method, fastNLP cannot save the training '
                               'state.')

        # 2. 保存模型的状态；
        if should_save_model:
            if not os.path.exists(folder):
                os.mkdir(folder)
            model_path = folder.joinpath(FASTNLP_MODEL_FILENAME)
            self.save_model(model_path, only_state_dict=only_state_dict)

        # 3. 保存 optimizers 的状态；
        states["optimizers_state_dict"] = self.get_optimizer_state()

        # 4. 保存fp16的状态

        logger.debug("Save optimizer state dict")
        jt.save(states, Path(folder).joinpath(FASTNLP_CHECKPOINT_FILENAME))

    def get_optimizer_state(self):
        optimizers_state_dict = {}
        for i in range(len(self.optimizers)):
            optimizer: Optimizer = self.optimizers[i]
            optimizers_state_dict[f"optimizer{i}"] = optimizer.state_dict()  # 注意这里没有使用 deepcopy，测试是不需要的；
        return optimizers_state_dict

    def load_optimizer_state(self, states):
        assert len(states) == len(self.optimizers), f"The number of optimizers is:{len(self.optimizers)}, while in " \
                                                    f"checkpoint it is:{len(states)}"
        for i in range(len(self.optimizers)):
            optimizer: Optimizer = self.optimizers[i]
            optimizer.load_state_dict(states[f"optimizer{i}"])
        logger.debug("Load optimizer state dict.")

    def load_checkpoint(self, folder: Path, dataloader, only_state_dict: bool = True, should_load_model: bool = True, **kwargs) -> Dict:
        
        states = jt.load(str(folder.joinpath(FASTNLP_CHECKPOINT_FILENAME)))

        # 1. 加载 optimizers 的状态；
        optimizers_state_dict = states.pop("optimizers_state_dict")
        self.load_optimizer_state(optimizers_state_dict)

        # 2. 加载模型状态；
        if should_load_model:
            self.load_model(filepath=folder.joinpath(FASTNLP_MODEL_FILENAME), only_state_dict=only_state_dict)

        # 3. 加载fp16的状态

        # 4. 恢复 sampler 的状态；
        dataloader_args = self.get_dataloader_args(dataloader)
        if dataloader_args.sampler is None:
            sampler = RandomSampler(dataloader_args.sampler.dataset, shuffle=dataloader_args.shuffle)
        elif isinstance(dataloader_args.sampler, ReproducibleSampler):
            sampler = dataloader_args.sampler
        elif isinstance(dataloader_args.sampler, JittorRandomSampler):
            sampler = RandomSampler(dataloader_args.sampler.dataset)
            logger.debug("Replace jittor RandomSampler into fastNLP RandomSampler.")
        elif isinstance(dataloader_args.sampler, JittorSequentialSampler):
            sampler = RandomSampler(dataloader_args.sampler.dataset, shuffle=False)
            logger.debug("Replace jittor Sampler into fastNLP RandomSampler without shuffle.")
        elif self.is_distributed():
            raise RuntimeError("It is not allowed to use checkpoint retraining when you do not use our"
                               "`ReproducibleSampler`.")
        else:
            raise RuntimeError(f"Jittor sampler {type(dataloader_args.sampler)} is not supported now.")
        sampler.load_state_dict(states.pop('sampler_states'))
        states["dataloader"] = self.set_dist_repro_dataloader(dataloader, sampler)

        # 4. 修改 trainer_state.batch_idx_in_epoch
        # sampler 是类似 RandomSampler 的sampler，不是 batch_sampler；
        if dataloader_args.drop_last:
            batch_idx_in_epoch = len(
                sampler) // dataloader_args.batch_size - sampler.num_left_samples // dataloader_args.batch_size
        else:
            batch_idx_in_epoch = (len(sampler) + dataloader_args.batch_size - 1) // dataloader_args.batch_size - \
                (sampler.num_left_samples + dataloader_args.batch_size - 1) // dataloader_args.batch_size

        states["batch_idx_in_epoch"] = batch_idx_in_epoch

        return states

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

    def set_deterministic_dataloader(self, dataloader: Union["JittorDataLoader", "Dataset"]):
        ...

    def set_sampler_epoch(self, dataloader: Union["JittorDataLoader", "Dataset"], cur_epoch_idx: int):
        # 保证 ddp 训练时的 shuffle=True 时的正确性，因为需要保证每一个进程上的 sampler 的shuffle 的随机数种子是一样的；
        if callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(cur_epoch_idx)

    @staticmethod
    def get_dataloader_args(dataloader: Union["JittorDataLoader", "Dataset"]):
        @dataclass
        class Res:
            dataset: Optional[Dataset] = None
            batch_sampler: Optional[JittorBatchSampler] = None
            sampler: Optional[JittorSampler] = None
            batch_size: Optional[int] = None
            shuffle: Optional[bool] = None
            drop_last: Optional[bool] = None

        res = Res()
        from fastNLP.core.dataloaders.jittor_dataloader.fdl import _JittorDataset
        if isinstance(dataloader, JittorDataLoader):
            # JittorDataLoader 实际上是迭代 dataset 成员的
            dataloader = dataloader.dataset
        if isinstance(dataloader, _JittorDataset):
            # 获取最原始的 dataset
            res.dataset = dataloader.dataset
        else:
            res.dataset = dataloader

        # jittor 现在不支持 batch_sampler，所以除了 shuffle 都可以直接获取
        res.batch_size = dataloader.batch_size
        res.drop_last = dataloader.drop_last
        if dataloader.sampler is None:
            # sampler 是 None，那么就从 Dataset 的属性中获取
            res.shuffle = dataloader.shuffle
        elif isinstance(list(dataloader.sampler.__iter__())[0], (list,tuple)):
            # jittor 目前不支持 batch_sampler
            raise NotImplementedError("Jittor does not support using batch_sampler in `Dataset` now, "
                                    "please check if you have set `Dataset.sampler` as `BatchSampler`")
        else:
            # sampler 不为 None
            res.sampler = dataloader.sampler
            if hasattr(dataloader.sampler, "shuffle"):
                # 这种情况一般出现在 fastNLP 的 ReproduceSampler 中
                res.shuffle = dataloader.sampler.shuffle
            elif isinstance(dataloader.sampler, JittorRandomSampler):
                res.shuffle = True
            else:
                res.shuffle = False

        return res