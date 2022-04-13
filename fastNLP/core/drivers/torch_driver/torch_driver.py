import os
from typing import Union, Dict, Optional, Callable
from functools import partial
from pkg_resources import parse_version
import numpy as np
import random
from dataclasses import dataclass
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from pathlib import Path
if _NEED_IMPORT_TORCH:
    import torch
    from torch.utils.data import DataLoader, IterableDataset, RandomSampler, Sampler, BatchSampler, Dataset
    from torch.optim import Optimizer
    _reduces = {
        'sum': torch.max,
        'min': torch.min,
        'max': torch.max,
        'mean': torch.mean
    }


__all__ = [
    'TorchDriver'
]

from .utils import optimizer_state_to_device
from fastNLP.core.drivers.driver import Driver
from fastNLP.core.drivers.torch_driver.utils import _build_fp16_env
from fastNLP.core.utils import apply_to_collection, torch_move_data_to_device
from fastNLP.envs import  rank_zero_call
from fastNLP.envs import FASTNLP_SEED_WORKERS, FASTNLP_GLOBAL_RANK, FASTNLP_MODEL_FILENAME, FASTNLP_CHECKPOINT_FILENAME
from fastNLP.core.log import logger
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleSampler, RandomBatchSampler


class TorchDriver(Driver):
    r"""
    专属于 pytorch 的 driver；因为我们会在同一个 Trainer 框架下提供 jittor、paddle 等训练框架的支持；
    """
    def __init__(self, model, fp16: Optional[bool] = False, **kwargs):
        super(TorchDriver, self).__init__(model)

        """ 进行 fp16 的设置 """
        # 因为 ddp 和 single_device 的混合精度训练的设置是一样的，因此可以统一抽象到这里；
        self.fp16 = fp16
        if parse_version(torch.__version__) < parse_version('1.6'):
            raise RuntimeError("Pytorch supports float16 after version 1.6, please upgrade your pytorch version.")
        self.auto_cast, _grad_scaler = _build_fp16_env(dummy=not fp16)
        self.grad_scaler = _grad_scaler()

        # 用来设置 `torch_move_data_to_device` 中的 `non_blocking` 参数；
        self.non_blocking = kwargs.get("torch_non_blocking", True)

        # 用来设置是否关闭 auto_param_call 中的参数匹配问题；
        self.wo_auto_param_call = kwargs.get("model_wo_auto_param_call", False)

    def zero_grad(self, set_to_none: bool = False):
        for optimizer in self.optimizers:
            self._clear_grad(optimizer, set_to_none)

    def _clear_grad(self, optimizer, set_to_none):
        param_groups = optimizer.param_groups
        for group in param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

    @staticmethod
    def _check_dataloader_legality(dataloader, dataloader_name, is_train: bool = False):
        if is_train:
            if not isinstance(dataloader, DataLoader):
                raise ValueError(f"Parameter `{dataloader_name}` should be 'DataLoader' type, not {type(dataloader)}.")

            # todo 我们先禁止 dataloader 的 dataset 是 IterableDataset 种类；
            if isinstance(dataloader.dataset, IterableDataset):
                raise TypeError("`IterableDataset` is not allowed.")

        else:
            if not isinstance(dataloader, Dict):
                raise ValueError(f"Parameter `{dataloader_name}` should be 'Dict' type, not {type(dataloader)}.")
            else:
                for each_dataloader in dataloader.values():
                    if not isinstance(each_dataloader, DataLoader):
                        raise ValueError(f"Each dataloader of parameter `{dataloader_name}` should be 'DataLoader' "
                                         f"type, not {type(each_dataloader)}.")
                    if isinstance(each_dataloader.dataset, IterableDataset):
                        raise TypeError("`IterableDataset` is not allowed.")

    @staticmethod
    def _check_optimizer_legality(optimizers):
        for each_optimizer in optimizers:
            if not isinstance(each_optimizer, Optimizer):
                raise ValueError(f"Each optimizer of parameter `optimizers` should be 'Optimizer' type, "
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

    @staticmethod
    def tensor_to_numeric(tensor, reduce=None):
        if tensor is None:
            return None

        def _translate(_data):
            if _data.numel() == 1:
                return _data.item()
            if reduce is None:
                return _data.tolist()
            return _reduces[reduce](_data).item()

        return apply_to_collection(
            data=tensor,
            dtype=torch.Tensor,
            function=_translate
        )

    def set_model_mode(self, mode: str):
        assert mode in {"train", "eval"}
        getattr(self.model, mode)()

    @rank_zero_call
    def save_model(self, filepath: Union[str, Path], only_state_dict: bool = True, **kwargs):
        """
        保存当前 driver 的模型到 folder 下。

        :param filepath: 保存到哪个文件夹；
        :param only_state_dict: 是否只保存权重；
        :return:
        """
        model = self.unwrap_model()

        if only_state_dict:
            states = {name: param.cpu().detach().clone() for name, param in model.state_dict().items()}
            torch.save(states, filepath)
        else:
            if self.model_device is not None:
                if not self.is_distributed():
                    self.move_model_to_device(model, torch.device("cpu"))
                torch.save(model, filepath)
                if not self.is_distributed():
                    self.move_model_to_device(model, self.model_device)
            else:
                torch.save(model, filepath)

    def load_model(self, filepath: str, only_state_dict: bool = True, **kwargs):
        """
        从 folder 中加载权重并赋值到当前 driver 的模型上。

        :param filepath: 加载权重或模型的路径
        :param load_state_dict: 保存的内容是否只是权重。
        :param kwargs:
        :return:
        """
        model = self.unwrap_model()
        res = torch.load(filepath, map_location='cpu')
        if only_state_dict:
            model.load_state_dict(res)
        else:
            model.load_state_dict(res.state_dict())

    @rank_zero_call
    def save(self, folder: Path, states: Dict, dataloader, only_state_dict: bool = True, should_save_model: bool = True, **kwargs):
        # 传入的 dataloader 参数是 trainer 的 dataloader 属性，因为 driver 的所有 dataloader 我们是不会去改变它的，而是通过改变
        #  trainer.dataloader 来改变 dataloader 的状态，从而适配训练或者评测环境；

        # 1. sampler 的状态，因为我们支持 resume training，即精确恢复到具体的一个 batch；
        # 首先 pytorch 的 DataLoader 一定会有 sampler；另一方面，我们在断点重训的时候一定会在 `set_` 中将 dataloader 的
        #  sampler 替换为 `ReproducibleSampler`；否则就是在单卡情况下将 batch_sampler 替换为 `ReproducibleBatchSampler`；
        dataloader_args = self.get_dataloader_args(dataloader)
        if isinstance(dataloader_args.batch_sampler, ReproducibleBatchSampler):
            sampler = dataloader_args.batch_sampler
        elif dataloader_args.sampler:
            sampler = dataloader_args.sampler
        else:
            raise RuntimeError("This condition is not supposed to appear. Please report a bug to us.")
        num_consumed_batches = states.pop('num_consumed_batches')
        if hasattr(sampler, 'state_dict') and callable(sampler.state_dict):
            sampler_states = sampler.state_dict()
            # 如果有，需要针对 num_consumed_samples 做特殊的处理。因为DataLoader存在预取行为，直接使用sampler中的num_consumed_samples
            #   会造成多余实际消耗的问题。
            num_consumed_samples_array = sampler_states.pop('num_consumed_samples_array', None)
            if num_consumed_samples_array is not None:
                if isinstance(sampler, ReproducibleSampler):  # 如果是 sampler 的话，需要考虑 batch_size 。
                    try:
                        num_consumed_batches = num_consumed_batches * dataloader_args.batch_size
                    except:  # 有可能 batch_size 为 None，就只有损失精度了
                        num_consumed_batches = sampler_states['num_consumed_samples']
                sampler_states['num_consumed_samples'] = num_consumed_samples_array[num_consumed_batches]
                assert sampler_states['num_consumed_samples'] != -1, "This is a bug, please report."
        else:
            raise RuntimeError(
                'The sampler has no `state_dict()` method, it will fail to recover to the specific batch.')

        # 2. 保存模型的状态；
        if should_save_model:
            model = self.unwrap_model()
            if only_state_dict:
                model_state_dict = {name: param.cpu().detach().clone() for name, param in model.state_dict().items()}
                # 对于单卡的 driver 来讲，我们实际上（现在）不应该考虑用户在DDP环境下使用单卡模式，从而造成效率损失；
                torch.save(model_state_dict, folder.joinpath(FASTNLP_MODEL_FILENAME))
                logger.debug("Save model state dict")
            else:
                torch.save(model, folder.joinpath(FASTNLP_MODEL_FILENAME))
                logger.debug("Save model")

        # 3. 保存 optimizers 的状态；
        optimizers_state_dict = {}
        for i in range(len(self.optimizers)):
            optimizer: torch.optim.Optimizer = self.optimizers[i]
            optimizer_state = optimizer.state_dict()
            optimizer_state["state"] = optimizer_state_to_device(optimizer_state["state"], torch.device("cpu"))
            optimizers_state_dict[f"optimizer{i}"] = optimizer_state  # 注意这里没有使用 deepcopy，测试是不需要的；

        logger.debug("Save optimizer state dict")
        states["optimizers_state_dict"] = optimizers_state_dict
        torch.save(states, Path(folder).joinpath(FASTNLP_CHECKPOINT_FILENAME))

    def load(self, folder: Path, dataloader, only_state_dict: bool = True, should_load_model: bool = True, **kwargs) -> Dict:
        states = torch.load(folder.joinpath(FASTNLP_CHECKPOINT_FILENAME))

        # 1. 加载 optimizers 的状态；
        optimizers_state_dict = states["optimizers_state_dict"]
        for i in range(len(self.optimizers)):
            optimizer: torch.optim.Optimizer = self.optimizers[i]
            optimizer.load_state_dict(optimizers_state_dict[f"optimizer{i}"])
        logger.debug("Load optimizer state dict.")

        # 2. 加载模型状态；
        if should_load_model:
            model = self.unwrap_model()
            res = torch.load(folder.joinpath(FASTNLP_MODEL_FILENAME), map_location='cpu')
            if only_state_dict:
                model.load_state_dict(res)
                logger.debug("Load model state dict.")
            else:
                model.load_state_dict(res.state_dict())
                logger.debug("Load model.")

        # 3. 恢复 sampler 的状态；
        dataloader_args = self.get_dataloader_args(dataloader)
        if isinstance(dataloader_args.batch_sampler, ReproducibleBatchSampler):
            sampler = dataloader_args.batch_sampler
        elif isinstance(dataloader_args.sampler, ReproducibleSampler):
            sampler = dataloader_args.sampler
        elif self.is_distributed():
            raise RuntimeError("It is not allowed to use checkpoint retraining when you do not use our or `ReproducibleSampler`.")
        else:
            sampler = RandomBatchSampler(
                batch_sampler=dataloader_args.batch_sampler if dataloader_args.batch_sampler is not None else dataloader_args.sampler,
                batch_size=dataloader_args.batch_size,
                drop_last=dataloader_args.drop_last
            )
        sampler.load_state_dict(states['sampler_states'])
        states["dataloader"] = self.set_dist_repro_dataloader(dataloader, sampler)

        # 4. 修改 trainer_state.batch_idx_in_epoch
        # sampler 是类似 RandomSampler 的sampler，不是 batch_sampler；
        if not isinstance(sampler, ReproducibleBatchSampler):
            if dataloader_args.drop_last:
                batch_idx_in_epoch = len(
                    sampler) // dataloader_args.batch_size - sampler.num_left_samples // dataloader_args.batch_size
            else:
                batch_idx_in_epoch = (len(sampler) + dataloader_args.batch_size - 1) // dataloader_args.batch_size - \
                    (sampler.num_left_samples + dataloader_args.batch_size - 1) // dataloader_args.batch_size
        # sampler 是 batch_sampler；
        else:
            batch_idx_in_epoch = sampler.batch_idx_in_epoch

        states["batch_idx_in_epoch"] = batch_idx_in_epoch

        return states

    def get_evaluate_context(self):
        return torch.no_grad

    @staticmethod
    def move_model_to_device(model: "torch.nn.Module", device: "torch.device"):
        if device is not None:
            model.to(device)

    def move_data_to_device(self, batch: "torch.Tensor"):
        return torch_move_data_to_device(batch, self.data_device, self.non_blocking)

    @staticmethod
    def worker_init_function(worker_id: int, rank: Optional[int] = None) -> None:  # pragma: no cover
        """The worker_init_fn that Lightning automatically adds to your dataloader if you previously set the seed
        with ``seed_everything(seed, workers=True)``.

        See also the PyTorch documentation on
        `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
        """
        # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
        global_rank = rank if rank is not None else int(os.environ.get(FASTNLP_GLOBAL_RANK, 0))
        process_seed = torch.initial_seed()
        # back out the base seed so we can use all the bits
        base_seed = process_seed - worker_id
        ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
        # use 128 bits (4 x 32-bit words)
        np.random.seed(ss.generate_state(4))
        # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
        torch_ss, stdlib_ss = ss.spawn(2)
        torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
        # use 128 bits expressed as an integer
        stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
        random.seed(stdlib_seed)

    def set_deterministic_dataloader(self, dataloader: "DataLoader"):
        if int(os.environ.get(FASTNLP_SEED_WORKERS, 0)) and dataloader.worker_init_fn is None:
            dataloader.worker_init_fn = partial(self.worker_init_function,
                                                rank=int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)))

    def set_sampler_epoch(self, dataloader: "DataLoader", cur_epoch_idx: int):
        # 保证 ddp 训练时的 shuffle=True 时的正确性，因为需要保证每一个进程上的 sampler 的shuffle 的随机数种子是一样的；
        if callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(cur_epoch_idx)

    @staticmethod
    def get_dataloader_args(dataloader: "DataLoader"):
        """
        获取 dataloader 的 shuffle 和 drop_last 属性；
        """

        @dataclass
        class Res:
            dataset: Optional[Dataset] = None
            batch_sampler: Optional[BatchSampler] = None
            sampler: Optional[Sampler] = None
            batch_size: Optional[int] = None
            shuffle: Optional[bool] = None
            drop_last: Optional[bool] = None

        res = Res()

        # pytorch 的 DataLoader 一定会有 dataset 属性；
        res.dataset = dataloader.dataset

        # dataloader 使用的是 sampler；
        if dataloader.batch_sampler is None:
            res.sampler = dataloader.sampler
            res.batch_size = 1
            res.shuffle = True if isinstance(dataloader.sampler, RandomSampler) else False
            res.drop_last = False
        # dataloader 使用的是 batch_sampler；
        else:
            res.batch_sampler = dataloader.batch_sampler
            if hasattr(dataloader.batch_sampler, "batch_size"):
                res.batch_size = getattr(dataloader.batch_sampler, "batch_size")
            # 用户使用的是自己的 batch_sampler 并且其没有 "batch_size" 属性；
            else:
                dataloader_iter = iter(dataloader)
                pre_sample = next(dataloader_iter)
                res.batch_size = pre_sample.shape[0]

            if hasattr(dataloader.batch_sampler, "sampler"):
                res.sampler = dataloader.batch_sampler.sampler
                if hasattr(dataloader.batch_sampler.sampler, "shuffle"):
                    res.shuffle = dataloader.batch_sampler.sampler.shuffle
                elif isinstance(dataloader.batch_sampler.sampler, RandomSampler):
                    res.shuffle = True
                else:
                    res.shuffle = False
            else:
                # 如果 dataloader.batch_sampler 没有 sampler 这个属性，那么说明其使用的是自己的 batch_sampler，且没有 "sampler" 属性；
                #  这种情况下 DataLoader 会自己初始化一个 sampler；我们因此将这个默认初始化的 sampler 挂载到 res 上；
                res.sampler = dataloader.sampler
                res.shuffle = False

            if hasattr(dataloader.batch_sampler, "drop_last"):
                res.drop_last = getattr(dataloader.batch_sampler, "drop_last")
            # 用户使用的是自己的 batch_sampler 并且其没有 "drop_last" 属性；
            else:
                res.drop_last = False

        return res
