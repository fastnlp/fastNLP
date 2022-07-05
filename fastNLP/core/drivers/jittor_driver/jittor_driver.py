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
    实现了 **jittor** 框架训练功能的基本 ``Driver``。这个类被以下子类继承：
    
        1. :class:`~fastNLP.core.drivers.jittor_driver.JittorSingleDriver` ：实现了使用单卡和 ``cpu`` 训练的具体功能；
        2. :class:`~fastNLP.core.drivers.jittor_driver.JittorMPIDriver` ：实现了使用 ``mpi`` 启动 **jittor** 分布式训练的功能；

    .. warning::

        您不应当直接初始化该类，然后传入给 ``Trainer``，换句话说，您应当使用该类的子类 ``JittorSingleDriver`` 和 ``TorchDDPDriver``，而不是
        该类本身。

    .. note::

        您可以在使用 ``JittorSingleDriver`` 和 ``JittorMPIDriver`` 时使用 ``JittorDriver`` 提供的接口。

    :param model: 训练时使用的 **jittor** 模型
    :param fp16: 是否开启混合精度训练
    :param jittor_kwargs:
    """
    def __init__(self, model, fp16: bool = False, jittor_kwargs: Dict = None, **kwargs):
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
        self._jittor_kwargs = jittor_kwargs if jittor_kwargs is not None else {}

        # 用来设置是否关闭 auto_param_call 中的参数匹配问题；
        self.wo_auto_param_call = kwargs.get("model_wo_auto_param_call", False)

    def check_dataloader_legality(self, dataloader):
        """
        检测 DataLoader 是否合法。支持的类型包括 :class:`~fastNLP.core.dataloaders.JittorDataLoader`、 :class:`jittor.dataset.Dataset` 。

        :param dataloder:
        """
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
        r"""
        实现参数的优化更新过程
        """
        for optimizer in self.optimizers:
            optimizer.step()

    def backward(self, loss):
        """
        对 ``loss`` 进行反向传播
        """        
        for optimizer in self.optimizers:
            optimizer.backward(loss)

    def zero_grad(self):
        """
        实现梯度置零的过程
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def save_model(self, filepath: Union[str, Path], only_state_dict: bool = True, **kwargs):
        r"""
        将模型保存到 ``filepath`` 中。

        :param filepath: 保存文件的文件位置
        :param only_state_dict: 在 **Jittor** 中，该参数无效，因为 **Jittor** 仅支持保存模型的 ``state_dict``。
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

        :param filepath: 保存文件的文件位置
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
        r"""
        断点重训的保存函数，该函数会负责保存 **优化器** 和 **sampler** 的状态，以及 **模型** （若 ``should_save_model`` 为 ``True``）

        :param folder: 保存断点重训的状态的文件夹；:meth:`save_checkpoint` 函数应该在该路径下面下面新增名为 ``FASTNLP_CHECKPOINT_FILENAME`` 与
            ``FASTNLP_MODEL_FILENAME`` （如果 ``should_save_model`` 为 ``True`` ）的文件。把 model 相关的内容放入到 ``FASTNLP_MODEL_FILENAME`` 文件
            中，将传入的 ``states`` 以及自身产生的其它状态一并保存在 ``FASTNLP_CHECKPOINT_FILENAME`` 里面。
        :param states: 由 :class:`~fastNLP.core.controllers.Trainer` 传入的一个字典，其中已经包含了为了实现断点重训所需要保存的其它对象的状态。
        :param dataloader: 正在使用的 dataloader。
        :param only_state_dict: 是否只保存模型的参数，当 ``should_save_model`` 为 ``False`` ，该参数无效。
        :param should_save_model: 是否应该保存模型，如果为 ``False`` ，Driver 将不负责 model 的保存。
        """
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
        r"""
        断点重训的加载函数，该函数会负责读取数据，并且恢复 **优化器** 、**sampler** 的状态和 **模型** （如果 ``should_load_model`` 为 True）以及其它
        在 :meth:`save_checkpoint` 函数中执行的保存操作，然后将一个 state 字典返回给 :class:`~fastNLP.core.controllers.Trainer` （ 内容为 :meth:`save_checkpoint` 
        接受到的 ``states`` ）。

        该函数应该在所有 rank 上执行。

        :param folder: 读取该 folder 下的 ``FASTNLP_CHECKPOINT_FILENAME`` 文件与 ``FASTNLP_MODEL_FILENAME``
            （如果 should_load_model 为True）。
        :param dataloader: 当前给定 dataloader，需要根据保存的 dataloader 状态合理设置。若该值为 ``None`` ，则不需要返回 ``'dataloader'``
            以及 ``'batch_idx_in_epoch'`` 这两个值。
        :param only_state_dict: 是否仅读取模型的 state_dict ，当 ``should_save_model`` 为 ``False`` ，该参数无效。如果为 ``True`` ，说明保存的内容为权重；如果为
            False 说明保存的是模型，但也是通过当前 Driver 的模型去加载保存的模型的权重，而不是使用保存的模型替换当前模型。
        :param should_load_model: 是否应该加载模型，如果为 ``False`` ，Driver 将不负责加载模型。若该参数为 ``True`` ，但在保存的状态中没有
            找到对应的模型状态，则报错。
        :return: :meth:`save_checkpoint` 函数输入的 ``states`` 内容。除此之外，还返回的内容有：

            * *dataloader* -- 根据传入的 ``dataloader`` 与读取出的状态设置为合理状态的 dataloader。在当前 ``dataloader`` 样本数与读取出的 sampler 样本数
              不一致时报错。
            * *batch_idx_in_epoch* -- :class:`int` 类型的数据，表明当前 epoch 进行到了第几个 batch 。请注意，该值不能仅通过保存的数据中读取的，因为前后两次运行的
              ``batch_size`` 可能有变化，而应该符合以下等式::

                返回的 dataloader 还会产生的 batch 数量 + batch_idx_in_epoch = 原来不断点训练时的 batch 的总数
              
              由于 ``返回的 dataloader 还会产生的batch数`` 在 ``batch_size`` 与 ``drop_last`` 参数给定的情况下，无法改变，因此只能通过调整 ``batch_idx_in_epoch``
              这个值来使等式成立。一个简单的计算原则如下：

                * drop_last 为 ``True`` 时，等同于 floor(sample_in_this_rank/batch_size) - floor(num_left_samples/batch_size)；
                * drop_last 为 ``False`` 时，等同于 ceil(sample_in_this_rank/batch_size) - ceil(num_left_samples/batch_size)。
        """
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
        r"""
        返回一个不计算梯度的上下文环境用来对模型进行评测；

        :return: 上下文对象 ``jittor.no_grad``
        """
        return jt.no_grad

    @staticmethod
    def move_model_to_device(model: "jt.Module", device):
        r"""
        将模型转移到指定的设备上。由于 **Jittor** 会自动为数据分配设备，因此该函数实际上无效。
        """
        ...

    def move_data_to_device(self, batch: 'jt.Var'):
        """
        将数据迁移到指定的机器上；**jittor** 会自动为变量分配设备无需手动迁移，因此这个函数只是简单地返回 ``batch``。
        """
        return batch

    def move_data_to_device(self, batch):
        """
        将数据 ``batch`` 转移到指定的设备上。由于 **Jittor** 会自动为数据分配设备，因此该函数实际上无效。
        """
        return batch

    @staticmethod
    def tensor_to_numeric(tensor, reduce=None):
        r"""
        将一个 :class:`jittor.Var` 对象转换为 转换成 python 中的数值类型。

        :param tensor: :class:`jittor.Var` 类型的对象
        :param reduce: 当 tensor 是一个多数值的张量时，应当使用何种归一化操作来转换成单一数值，应当为以下类型之一：``['max', 'min', 'sum', 'mean']``。
        :return: 返回一个单一数值，其数值类型是 python 中的基本的数值类型，例如 ``int，float`` 等
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
        r"""
        设置模型为 ``train`` 或 ``eval`` 的模式；目的是为切换模型的训练和推理（会关闭 dropout 等）模式。

        :param mode: 应为二者之一：``["train", "eval"]``
        """
        assert mode in {"train", "eval"}
        getattr(self.model, mode)()

    @property
    def data_device(self):
        """
        :return: 数据默认会被迁移到的设备
        """
        return self.model_device

    def set_deterministic_dataloader(self, dataloader: Union["JittorDataLoader", "Dataset"]):
        r"""
        为了确定性训练要对 ``dataloader`` 进行修改，保证在确定随机数种子后，每次重新训练得到的结果是一样的。 **jittor** 暂时不提供
        该功能。
        """
        ...

    def set_sampler_epoch(self, dataloader: Union["JittorDataLoader", "Dataset"], cur_epoch_idx: int):
        r"""
        对于分布式的 ``sampler``，需要在每一个 ``epoch`` 前设置随机数种子，来保证每一个进程上的 ``shuffle`` 是一样的。

        :param dataloader: 需要设置 ``epoch`` 的 ``dataloader``
        :param cur_epoch_idx: 当前是第几个 ``epoch``
        """
        # 保证 ddp 训练时的 shuffle=True 时的正确性，因为需要保证每一个进程上的 sampler 的shuffle 的随机数种子是一样的；
        if callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(cur_epoch_idx)

    @staticmethod
    def get_dataloader_args(dataloader: Union["JittorDataLoader", "Dataset"]):
        """
        从 ``dataloader`` 中获取参数 ``dataset``, ``batch_sampler``, ``sampler``, ``batch_size``, ``shuffle`` 
        和 ``drop_last`` 。
        """
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