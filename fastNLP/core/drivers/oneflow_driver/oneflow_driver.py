import os
from typing import Union, Dict, Optional, Callable, Tuple
from functools import partial
import numpy as np
import random
from dataclasses import dataclass
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW
from pathlib import Path
if _NEED_IMPORT_ONEFLOW:
    import oneflow
    from oneflow.utils.data import DataLoader, Sampler, BatchSampler, Dataset
    from oneflow.optim import Optimizer
    from oneflow.utils.data import RandomSampler as OneflowRandomSampler
    _reduces = {
        "sum": oneflow.sum,
        "min": oneflow.min,
        "max": oneflow.max,
        "mean": oneflow.mean
    }


__all__ = [
    "OneflowDriver"
]

from .utils import optimizer_state_to_device, DummyGradScaler
from fastNLP.core.drivers.driver import Driver
from fastNLP.core.utils.utils import _get_fun_msg, nullcontext
from fastNLP.core.utils import apply_to_collection, oneflow_move_data_to_device, auto_param_call
from fastNLP.envs import  rank_zero_call
from fastNLP.envs import FASTNLP_GLOBAL_RANK, FASTNLP_MODEL_FILENAME, FASTNLP_CHECKPOINT_FILENAME
from fastNLP.core.log import logger
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleSampler, ReproduceBatchSampler, RandomSampler
from fastNLP.core.dataloaders import OverfitDataLoader


class OneflowDriver(Driver):
    r"""
    专属于 ``oneflow`` 的 ``driver``，是 ``OneflowSingleDriver`` 和 ``OneflowDDPDriver`` 的父类；

    .. warning::

        您不应当直接初始化该类，然后传入给 ``Trainer``，换句话说，您应当使用该类的子类 ``OneflowSingleDriver`` 和 ``OneflowDDPDriver``，而不是
        该类本身；

    .. note::

        您可以在使用 ``OneflowSingleDriver`` 和 ``OneflowDDPDriver`` 时使用 ``OneflowDriver`` 提供的接口；

    """
    def __init__(self, model, fp16: Optional[bool] = False, oneflow_kwargs: Dict = {}, **kwargs):
        super(OneflowDriver, self).__init__(model)

        """ 进行 fp16 的设置 """
        self._oneflow_kwargs = oneflow_kwargs

        self.fp16 = fp16
        if fp16:
            logger.warn("OneflowDriver of eager mode dose not support fp16 now.``")
        # self.auto_cast, _grad_scaler = _build_fp16_env(dummy=not self.fp16)
        # self.grad_scaler = _grad_scaler(**self._oneflow_kwargs.get("gradscaler_kwargs", {}))
        self.auto_cast = nullcontext
        self.grad_scaler = DummyGradScaler()
        self.set_grad_to_none = self._oneflow_kwargs.get("set_grad_to_none")

        self.wo_auto_param_call = kwargs.get("model_wo_auto_param_call", False)

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad(self.set_grad_to_none)

    def backward(self, loss):
        loss.backward()
        # self.grad_scaler.scale(loss).backward()

    def step(self):
        for optimizer in self.optimizers:
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()

    def check_dataloader_legality(self, dataloader):
        if not isinstance(dataloader, DataLoader) and not isinstance(dataloader, OverfitDataLoader):
            raise TypeError(f"{DataLoader} is expected, instead of `{type(dataloader)}`")
        if len(dataloader) == 0:
            logger.rank_zero_warning("Your dataloader is empty, which is not recommended because it "
                                        "may cause some unexpected exceptions.", once=True)

    @staticmethod
    def _check_optimizer_legality(optimizers):
        for each_optimizer in optimizers:
            if not isinstance(each_optimizer, Optimizer):
                raise TypeError(f"Each optimizer of parameter `optimizers` should be 'Optimizer' type, "
                                 f"not {type(each_optimizer)}.")

    @staticmethod
    def tensor_to_numeric(tensor, reduce: str = None):
        r"""
        将 ``oneflow.Tensor`` 转换成 python 中的数值类型；

        :param tensor: ``oneflow.Tensor``；
        :param reduce: 当 tensor 是一个多数值的张量时，应当使用何种归一化操作来转换成单一数值，应当为以下类型之一：``['max', 'min', 'sum', 'mean']``；
        :return: 返回一个单一数值，其数值类型是 python 中的基本的数值类型，例如 ``int，float`` 等；
        """

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
            dtype=oneflow.Tensor,
            function=_translate
        )

    def set_model_mode(self, mode: str):
        r"""
        设置模型的状态是 ``train`` 还是 ``eval``；
        :param mode: ``'train'`` 或 ``'eval'``；
        """
        assert mode in {"train", "eval"}
        getattr(self.model, mode)()

    @rank_zero_call
    def save_model(self, filepath: Union[str, Path], only_state_dict: bool = True, **kwargs):
        """
        保存当前 driver 的模型到 folder 下。

        :param filepath: 保存到哪个文件夹；
        :param only_state_dict: 是否只保存权重；如果使用 ``DistributedDataParallel`` 启动分布式训练的话，该参数只能为 ``True``；
        :return:
        """
        model = self.unwrap_model()
        if not only_state_dict and self.is_distributed():
            logger.warn("`Cannot save ddp model directly, we will save its state_dict for you.")
            only_state_dict = True

        if only_state_dict:
            states = {name: param.cpu().detach().clone() for name, param in model.state_dict().items()}
            oneflow.save(states, filepath)
        else:
            if self.model_device is not None:
                if not self.is_distributed():
                    self.move_model_to_device(model, oneflow.device("cpu"))
                oneflow.save(model, filepath)
                if not self.is_distributed():
                    self.move_model_to_device(model, self.model_device)
            else:
                oneflow.save(model, filepath)

    def load_model(self, filepath: Union[Path, str], only_state_dict: bool = True, **kwargs):
        """
        从 folder 中加载权重并赋值到当前 driver 的模型上。

        :param filepath: 加载权重或模型的路径
        :param load_state_dict: 保存的内容是否只是权重。
        :param kwargs:
        :return:
        """
        model = self.unwrap_model()
        res = oneflow.load(filepath)
        if isinstance(res, dict) and only_state_dict is False:
            logger.rank_zero_warning(f"It seems like that {filepath} only contains state, you may need to use "
                                     f"`only_state_dict=True`")
        elif not isinstance(res, dict) and only_state_dict is True:
            logger.rank_zero_warning(f"It seems like that {filepath} is not state, you may need to use "
                                     f"`only_state_dict=False`")
        if not isinstance(res, dict):
            res = res.state_dict()
        model.load_state_dict(res)

    @rank_zero_call
    def save_checkpoint(self, folder: Path, states: Dict, dataloader, only_state_dict: bool = True, should_save_model: bool = True, **kwargs):
        # 传入的 dataloader 参数是 trainer 的 dataloader 属性，因为 driver 的所有 dataloader 我们是不会去改变它的，而是通过改变
        #  trainer.dataloader 来改变 dataloader 的状态，从而适配训练或者评测环境；

        # 1. sampler 的状态；
        num_consumed_batches = states.pop("num_consumed_batches")
        states["sampler_states"] = self.get_sampler_state(dataloader, num_consumed_batches)

        # 2. 保存模型的状态；
        if should_save_model:
            if not os.path.exists(folder):
                os.mkdir(folder)
            model_path = folder.joinpath(FASTNLP_MODEL_FILENAME)
            self.save_model(model_path, only_state_dict=only_state_dict)

        # 3. 保存 optimizers 的状态；
        states["optimizers_state_dict"] = self.get_optimizer_state()
        logger.debug("Save optimizer state dict.")

        # # 4. 保存fp16的状态
        # if not isinstance(self.grad_scaler, DummyGradScaler):
        #     grad_scaler_state_dict = self.grad_scaler.state_dict()
        #     states['grad_scaler_state_dict'] = grad_scaler_state_dict

        oneflow.save(states, Path(folder).joinpath(FASTNLP_CHECKPOINT_FILENAME))

    def get_sampler_state(self, dataloader, num_consumed_batches):
        dataloader_args = self.get_dataloader_args(dataloader)
        if isinstance(dataloader_args.batch_sampler, ReproducibleBatchSampler):
            sampler = dataloader_args.batch_sampler
        elif dataloader_args.sampler:
            sampler = dataloader_args.sampler
        else:
            raise RuntimeError("This condition is not supposed to appear. Please report a bug to us.")

        if hasattr(sampler, "state_dict") and callable(sampler.state_dict):
            sampler_states = sampler.state_dict()
            if dataloader_args.batch_size is not None:
                sampler_states["num_consumed_samples"] = sampler.num_replicas * dataloader_args.batch_size \
                                                         * num_consumed_batches
            else:
                logger.rank_zero_warning("fastNLP cannot get batch_size, we have to save based on sampler's "
                                         "`num_consumed_samples`, it may cause missing some samples when reload.")
        else:
            raise RuntimeError("The sampler has no `state_dict()` method, fastNLP cannot save the training "
                               "state.")

        return sampler_states

    def load_sampler_state(self, dataloader, sampler_states):
        states = {}
        dataloader_args = self.get_dataloader_args(dataloader)
        if isinstance(dataloader_args.batch_sampler, ReproducibleBatchSampler):
            sampler = dataloader_args.batch_sampler
        elif isinstance(dataloader_args.sampler, ReproducibleSampler):
            sampler = dataloader_args.sampler
        elif isinstance(dataloader_args.sampler, OneflowRandomSampler):
            sampler = RandomSampler(dataloader_args.sampler.data_source)
            logger.debug("Replace oneflow RandomSampler into fastNLP RandomSampler.")
        elif self.is_distributed():
            raise RuntimeError("It is not allowed to use checkpoint retraining when you do not use our"
                               "`ReproducibleSampler`.")
        else:
            sampler = ReproduceBatchSampler(
                batch_sampler=dataloader_args.batch_sampler if dataloader_args.batch_sampler is not None else dataloader_args.sampler,
                batch_size=dataloader_args.batch_size,
                drop_last=dataloader_args.drop_last
            )
        sampler.load_state_dict(sampler_states)
        states["dataloader"] = self.set_dist_repro_dataloader(dataloader, sampler)

        # 修改 trainer_state.batch_idx_in_epoch
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

    def get_optimizer_state(self):
        optimizers_state_dict = {}
        for i in range(len(self.optimizers)):
            optimizer: oneflow.optim.Optimizer = self.optimizers[i]
            optimizer_state = optimizer.state_dict()
            optimizer_state["state"] = optimizer_state_to_device(optimizer_state["state"], oneflow.device("cpu"))
            optimizers_state_dict[f"optimizer{i}"] = optimizer_state  # 注意这里没有使用 deepcopy，测试是不需要的；
        return optimizers_state_dict

    def load_optimizer_state(self, states):
        assert len(states) == len(self.optimizers), f"The number of optimizers is:{len(self.optimizers)}, while in " \
                                                    f"checkpoint it is:{len(states)}"
        for i in range(len(self.optimizers)):
            optimizer: oneflow.optim.Optimizer = self.optimizers[i]
            optimizer.load_state_dict(states[f"optimizer{i}"])
        logger.debug("Load optimizer state dict.")

    def load_checkpoint(self, folder: Path, dataloader, only_state_dict: bool = True, should_load_model: bool = True, **kwargs) -> Dict:
        states = oneflow.load(folder.joinpath(FASTNLP_CHECKPOINT_FILENAME))

        # 1. 加载 optimizers 的状态；
        optimizers_state_dict = states.pop("optimizers_state_dict")
        self.load_optimizer_state(optimizers_state_dict)

        # 2. 加载模型状态；
        if should_load_model:
            self.load_model(filepath=folder.joinpath(FASTNLP_MODEL_FILENAME), only_state_dict=only_state_dict)

        # # 3. 加载 fp16 的状态
        # if "grad_scaler_state_dict" in states:
        #     grad_scaler_state_dict = states.pop("grad_scaler_state_dict")
        #     if not isinstance(self.grad_scaler, DummyGradScaler):
        #         self.grad_scaler.load_state_dict(grad_scaler_state_dict)
        #         logger.debug("Load grad_scaler state dict...")
        # elif not isinstance(self.grad_scaler, DummyGradScaler):
        #     logger.rank_zero_warning(f"Checkpoint {folder} is not trained with fp16=True, while resume to a fp16=True training, "
        #                    f"the training process may be unstable.")

        # 4. 恢复 sampler 的状态；
        sampler_states = states.pop("sampler_states")
        states_ret = self.load_sampler_state(dataloader, sampler_states)
        states.update(states_ret)

        return states

    def get_evaluate_context(self):
        r"""
        :return: 返回 ``oneflow.no_grad`` 这个 context；
        """
        return oneflow.no_grad

    def model_call(self, batch, fn: Callable, signature_fn: Optional[Callable]) -> Dict:
        if isinstance(batch, Dict) and not self.wo_auto_param_call:
            return auto_param_call(fn, batch, signature_fn=signature_fn)
        else:
            return fn(batch)

    def get_model_call_fn(self, fn: str) -> Tuple:
        if hasattr(self.model, fn):
            fn = getattr(self.model, fn)
            if not callable(fn):
                raise RuntimeError(f"The `{fn}` attribute is not `Callable`.")
            logger.debug(f"Use {_get_fun_msg(fn, with_fp=False)}...")
            return fn, None
        elif fn in {"train_step", "evaluate_step"}:
            logger.debug(f"Use {_get_fun_msg(self.model.forward, with_fp=False)}...")
            return self.model, self.model.forward
        else:
            raise RuntimeError(f"There is no `{fn}` method in your {type(self.model)}.")

    @staticmethod
    def move_model_to_device(model: "oneflow.nn.Module", device: "oneflow.device"):
        r"""
        将模型迁移到对应的设备上；
        """
        if device is not None:
            model.to(device)

    def move_data_to_device(self, batch):
        """
        将一个 batch 的数据迁移到对应的设备上；

        :param batch: 一个 batch 的数据，可以是 ``list、dict`` 等；
        :return:
        """
        return oneflow_move_data_to_device(batch, self.data_device)

    @staticmethod
    def worker_init_function(worker_id: int, rank: Optional[int] = None) -> None:  # pragma: no cover
        global_rank = rank if rank is not None else int(os.environ.get(FASTNLP_GLOBAL_RANK, 0))
        process_seed = oneflow.initial_seed()

        base_seed = process_seed - worker_id
        ss = np.random.SeedSequence([base_seed, worker_id, global_rank])

        np.random.seed(ss.generate_state(4))

        oneflow_ss, stdlib_ss = ss.spawn(2)
        oneflow.manual_seed(oneflow_ss.generate_state(1, dtype=np.uint64)[0])

        stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
        random.seed(stdlib_seed)

    def set_deterministic_dataloader(self, dataloader: "DataLoader"):
        if dataloader.worker_init_fn is None:
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

        # oneflow 的 DataLoader 一定会有 dataset 属性；
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
                elif isinstance(dataloader.batch_sampler.sampler, OneflowRandomSampler):
                    res.shuffle = True
                else:
                    res.shuffle = False
            # ReproduceBatchSampler 的情况
            elif hasattr(dataloader.batch_sampler, "batch_sampler"):
                batch_sampler = dataloader.batch_sampler.batch_sampler
                res.sampler = batch_sampler.sampler
                if hasattr(batch_sampler.sampler, "shuffle"):
                    res.shuffle = dataloader.batch_sampler.sampler.shuffle
                elif isinstance(batch_sampler.sampler, OneflowRandomSampler):
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
