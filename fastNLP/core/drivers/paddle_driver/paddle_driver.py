import os
import random
from typing import Union, Optional, Dict
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import numpy as np

from .utils import _build_fp16_env, optimizer_state_to_device, DummyGradScaler
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.core.drivers.driver import Driver
from fastNLP.core.utils import apply_to_collection, paddle_move_data_to_device, get_device_from_visible
from fastNLP.envs import (
    FASTNLP_SEED_WORKERS,
    FASTNLP_MODEL_FILENAME,
    FASTNLP_CHECKPOINT_FILENAME,
    FASTNLP_GLOBAL_RANK,
    rank_zero_call,
)
from fastNLP.core.log import logger
from fastNLP.core.samplers import (
    ReproducibleBatchSampler,
    ReproducibleSampler,
    ReproduceBatchSampler,
    RandomSampler,
)

if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle.io import (
        DataLoader,
        IterableDataset,
        Dataset,
        Sampler,
        BatchSampler,
        RandomSampler as PaddleRandomSampler,
    )
    from paddle.optimizer import Optimizer

    _reduces = {
        "max": paddle.max,
        "min": paddle.min,
        "mean": paddle.mean,
        "sum": paddle.sum
    }

class PaddleDriver(Driver):
    r"""
    Paddle框架的Driver，包括实现单卡训练的`PaddleSingleDriver`和分布式训练的`PaddleFleetDriver`。
    """
    def __init__(self, model, fp16: Optional[bool] = False, **kwargs):
        if not isinstance(model, paddle.nn.Layer):
            raise ValueError(f"Parameter `model` can not be `{type(model)}` in `PaddleDriver`, it should be exactly "
                            f"`paddle.nn.Layer` type.")

        super(PaddleDriver, self).__init__(model)
        self.fp16 = fp16

        # scaler的参数
        self.auto_cast, _grad_scaler = _build_fp16_env(dummy=not fp16)
        self.grad_scaler = _grad_scaler()

        # 用来设置是否关闭 auto_param_call 中的参数匹配问题；
        self.wo_auto_param_call = kwargs.get("model_wo_auto_param_call", False)

    def zero_grad(self, set_to_none: bool = False):
        r"""
        实现深度学习中的梯度的置零操作，应当直接通过优化器 optimizers 来将梯度置零；
        注意梯度累积不需要在这里实现，trainer 已经在内部实现了梯度累积；

        :param set_to_none: 用来判断是否需要将梯度直接置为 None；Paddle中这个参数无效。
        """
        if set_to_none:
            logger.rank_zero_warning("Parameter `set_to_none` does nothing in paddle since grad cannot be set directly.")
        for optimizer in self.optimizers:
            optimizer.clear_grad()

    def backward(self, loss):
        self.grad_scaler.scale(loss).backward()

    def step(self):
        for optimizer in self.optimizers:
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()

    @staticmethod
    def check_dataloader_legality(dataloader, dataloader_name, is_train: bool = False):
        r"""
        该函数会在 trainer 或者 evaluator 设置 dataloader 后检测 dataloader 的合法性。
        要求传入的 dataloader 必须为 `paddle.io.DataLoader` 或包含该类型的字典。

        :param dataloader: 需要检测的输入的 `dataloader`；
        :param dataloader_name: 
        :param is_train:
        """
        if is_train:
            if not isinstance(dataloader, DataLoader):
                raise ValueError(f"Parameter `{dataloader_name}` should be 'paddle.io.DataLoader' type, not {type(dataloader)}.")
            # TODO 我们先禁止 dataloader 的 dataset 是 IterableDataset 种类；
            if isinstance(dataloader.dataset, IterableDataset):
                raise TypeError("`IterableDataset` is not allowed.")
            if dataloader.batch_sampler is None and dataloader.batch_size is None:
                raise ValueError(f"At least one of `{dataloader_name}`'s `batch_sampler` and `batch_size` should be set.")
        else:
            if not isinstance(dataloader, Dict):
                raise ValueError(f"Parameter `{dataloader_name}` should be 'Dict' type, not {type(dataloader)}.")
            else:
                for each_dataloader in dataloader.values():
                    if not isinstance(each_dataloader, DataLoader):
                        raise ValueError(f"Each dataloader of parameter `{dataloader_name}` should be 'paddle.io.DataLoader' "
                                         f"type, not {type(each_dataloader)}.")
                    if isinstance(each_dataloader.dataset, IterableDataset):
                        raise TypeError("`IterableDataset` is not allowed.")
                    if each_dataloader.batch_sampler is None and each_dataloader.batch_size is None:
                        raise ValueError(f"For each dataloader of parameter `{dataloader_name}`, at least one of "
                                         f"`batch_sampler` and `batch_size` should be set.")

    @staticmethod
    def _check_optimizer_legality(optimizers):
        r"""
        对于用户传入 trainer 的每一个 optimizer检测其合法性，必须为`paddle.optimizer.Optimizer`类型。

        :param optimizers: 需要检测的 `optimizers`；
        """
        for each_optimizer in optimizers:
            if not isinstance(each_optimizer, Optimizer):
                raise ValueError(f"Each optimizer of parameter `optimizers` should be 'paddle.optimizer.Optimizer' type, "
                                 f"not {type(each_optimizer)}.")

    @staticmethod
    def tensor_to_numeric(tensor, reduce=None):
        r"""
        将一个 `tensor` 对象（类型为 `paddle.Tensor` ）转换为 python 的 `numeric` 对象；如果 tensor 只包含一个
            元素则返回 float 或 int 。

        :param tensor: 需要被转换的 `tensor` 对象
        :param reduce: 可选 ['sum', 'max', 'mea', 'min']，如果不为 None 将使用该 reduce 方法来处理当前 tensor 再返回
            float 或 int 对象。
        :return: 转换后返回的结果
        """
        if tensor is None:
            return None

        def _translate(_data):
            # 如果只含有一个元素，则返回元素本身，而非list
            if _data.numel().item() == 1:
                return _data.item()
            if reduce is None:
                return _data.tolist()
            else:
                return _reduces[reduce](_data).item()

        return apply_to_collection(
            data=tensor,
            dtype=paddle.Tensor,
            function=_translate
        )

    def set_model_mode(self, mode: str):
        r"""
        设置模型为 `train` / `eval` 的模式；目的是为切换模型训练和推理（会关闭dropout等）模式；

        :param mode: 应为二者之一：["train", "eval"]；
        """
        assert mode in {"train", "eval"}
        getattr(self.model, mode)()

    @rank_zero_call
    def save_model(self, filepath: str, only_state_dict: bool = True, **kwargs):
        r"""
        保存模型的函数；注意函数 `save` 是用来进行断点重训的函数；

        :param filepath: 保存文件的文件位置（需要包括文件名）；
        :param only_state_dict: 是否只保存模型的 `state_dict`；如果为 False，则会调用 `paddle.jit.save` 函数
                                保存整个模型的参数，此时需要传入 `input_spec` 参数，否则在 load 时会报错。
        :param kwargs:
                input_spec: 描述存储模型 forward 方法的输入，当 `only_state_dict` 为 False时必须传入，否则加载时会报错。
                            可以通过 InputSpec 或者示例 Tensor 进行描述。详细的可以参考 paddle 关于`paddle.jit.save`
                            的文档：
                            https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/save_cn.html#save
        :return:
        """
        model = self.unwrap_model()
        if isinstance(filepath, Path):
            filepath = str(filepath)
        if only_state_dict:
            states = {name: param.cpu().detach().clone() for name, param in model.state_dict().items()}
            paddle.save(states, filepath)
        else:
            # paddle 在保存整个模型时需要传入额外参数
            input_spec = kwargs.get("input_spec", None)
            if input_spec is None:
                raise ValueError("To save the whole Paddle Layer, parameter `input_spec` is needed.")
            paddle.jit.save(model, filepath, input_spec)

    def load_model(self, filepath: str, only_state_dict: bool = True, **kwargs):
        r"""
        加载模型的函数；注意函数 `load` 是用来进行断点重训的函数；

        :param filepath: 需要被加载的对象的文件位置（需要包括文件名）；
        :param only_state_dict: 是否加载state_dict，默认为True。
        :param kwargs:
        :return:
        """
        model = self.unwrap_model()
        if isinstance(filepath, Path):
            filepath = str(filepath)
        # paddle 中，通过 paddle.jit.save 函数保存的模型也可以通过 paddle.load 加载为相应的 state dict
        # 但是此时对输入的 path 有要求，必须是 dir/filename 的形式，否则会报错。
        dirname, filename = os.path.split(filepath)
        if not only_state_dict and dirname == "":
            # 如果传入的是单个文件，则加上相对路径
            filepath = os.path.join(".", filepath)
        model.load_dict(paddle.load(filepath))

    @rank_zero_call
    def save(self, folder: Path, states: Dict, dataloader, only_state_dict: bool = True, should_save_model: bool = True, **kwargs):
        r"""
        断点重训的保存函数，该函数会负责保存模型和 optimizers 的 state_dict；
        需要注意 driver 应当是无状态的，即不管什么时候调用 driver 的接口函数，其返回的结果应该都是一样的；因此，断点重训不需要保存 driver
         本身自己的任何状态；而每一个 driver 实例需要在该函数中实现保存模型和 optimizers 的 state_dict 的逻辑；同时妥善存储传入的
         states 中的内容（主要用于恢复 Trainer ，Callback 等）
        需要保证该函数只在 global rank 0 上运行

        :param folder: 保存断点重训的状态的文件名；
        :param states: 由 trainer 传入的一个字典，其中已经包含了为了实现断点重训所需要保存的其它对象的状态，Driver 应该只需要保存
            该对象即可， Driver 应该不需要理解该对象，同时在 driver.load() 的时候，需要将 states 返回回去，load()返回的值与这里的
            传入的值保持一致。
        :param dataloader: 正在使用的 dataloader，需要保存里面的状态使得之后可以从当前迭代的位置恢复。
        :param only_state_dict: 是否只保存模型的参数，当 should_save_model 为 False ，该参数无效。
        :param should_save_model: 是否应该保存模型，如果为False，Driver 将不负责 model 的保存。
        :return:
        """
        # 传入的 dataloader 参数是 trainer 的 dataloader 属性，因为 driver 的所有 dataloader 我们是不会去改变它的，而是通过改变
        #  trainer.dataloader 来改变 dataloader 的状态，从而适配训练或者评测环境；

        # 1. sampler 的状态，因为我们支持 resume training，即精确恢复到具体的一个 batch；
        # paddle 的 DataLoader 在初始化之后 batch_sampler 可能为 None，也可能为用户设置的 batch_sampler
        dataloader_args = self.get_dataloader_args(dataloader)
        if isinstance(dataloader_args.batch_sampler, ReproducibleBatchSampler):
            sampler = dataloader_args.batch_sampler
        elif dataloader_args.sampler:
            sampler = dataloader_args.sampler
        else:
            raise RuntimeError("This condition is not supposed to appear. Please report a bug to us.")

        num_consumed_batches = states.pop("num_consumed_batches")
        if hasattr(sampler, "state_dict") and callable(sampler.state_dict):
            sampler_states = sampler.state_dict()
            # 如果有，需要针对 num_consumed_samples 做特殊的处理。因为DataLoader存在预取行为，直接使用sampler中的num_consumed_samples
            #  会造成多余实际消耗的问题。
            num_consumed_samples_array = sampler_states.pop('num_consumed_samples_array', None)
            if num_consumed_samples_array is not None:
                if isinstance(sampler, ReproducibleSampler):  # 如果是 sampler 的话，需要考虑 batch_size 。
                    if dataloader_args.batch_size is not None:
                        num_consumed_batches = num_consumed_batches * dataloader_args.batch_size
                    else:  # 有可能 batch_size 为 None，就只有损失精度了
                        logger.rank_zero_warning("fastNLP cannot get batch_size, we have to save based on `num_consumed_samples`, "
                                     "it may cause missing some samples when reload.")
                        num_consumed_batches = sampler_states['num_consumed_samples']
                sampler_states['num_consumed_samples'] = num_consumed_samples_array[num_consumed_batches]
                assert sampler_states['num_consumed_samples'] != -1, "This is a bug, please report."
            else:
                if dataloader_args.batch_size is not None:
                    sampler_states['num_consumed_samples'] = sampler.num_replicas * dataloader_args.batch_size \
                                                             * num_consumed_batches
                else:
                    logger.rank_zero_warning("fastNLP cannot get batch_size, we have to save based on `num_consumed_samples`, "
                                 "it may cause missing some samples when reload.")
        else:
            raise RuntimeError(
                "The sampler has no `state_dict()` method, it will fail to recover to the specific batch.")
        
        states['sampler_states'] = sampler_states

        # 2. 保存模型的状态；
        if should_save_model:
            self.save_model(folder.joinpath(FASTNLP_MODEL_FILENAME), only_state_dict, **kwargs)
            if only_state_dict:
                logger.debug("Save model state dict.")
            else:
                logger.debug("Save model.")

        # 3. 保存 optimizers 的状态；
        optimizers_state_dict = {}
        for i in range(len(self.optimizers)):
            optimizer: Optimizer = self.optimizers[i]
            optimizer_state = optimizer.state_dict()
            optimizer_state["state"] = optimizer_state_to_device(optimizer_state, "cpu")
            optimizers_state_dict[f"optimizer{i}"] = optimizer_state  # 注意这里没有使用 deepcopy，测试是不需要的；

        logger.debug("Save optimizer state dict.")
        states["optimizers_state_dict"] = optimizers_state_dict

        # 4.保存fp16的状态
        if not isinstance(self.grad_scaler, DummyGradScaler):
            grad_scaler_state_dict = self.grad_scaler.state_dict()
            states['grad_scaler_state_dict'] = grad_scaler_state_dict

        paddle.save(states, str(folder.joinpath(FASTNLP_CHECKPOINT_FILENAME)))

    def load(self, folder: Path, dataloader, only_state_dict: bool = True, should_load_model: bool = True, **kwargs) -> Dict:
        
        states = paddle.load(str(folder.joinpath(FASTNLP_CHECKPOINT_FILENAME)))

        # 1. 加载 optimizers 的状态；
        optimizers_state_dict = states.pop("optimizers_state_dict")
        for i in range(len(self.optimizers)):
            optimizer: Optimizer = self.optimizers[i]
            optimizer.set_state_dict(optimizers_state_dict[f"optimizer{i}"])
        logger.debug("Load optimizer state dict.")

        # 2. 加载模型状态；
        if should_load_model:
            self.load_model(folder.joinpath(FASTNLP_MODEL_FILENAME), only_state_dict)
            if only_state_dict:
                logger.debug("Load model state dict...")
            else:
                logger.debug("Load model...")

        # 3. 加载fp16的状态；
        if "grad_scaler_state_dict" in states:
            grad_scaler_state_dict = states.pop("grad_scaler_state_dict")
            if isinstance(self.grad_scaler, DummyGradScaler):
                self.auto_cast, _grad_scaler = _build_fp16_env(dummy=False)
                self.grad_scaler = _grad_scaler()
                self.fp16 = True
            self.grad_scaler.load_state_dict(grad_scaler_state_dict)
            logger.debug("Load grad_scaler state dict...")
        elif not isinstance(self.grad_scaler, DummyGradScaler):
            logger.rank_zero_warning(f"Checkpoint {folder} is not trained with fp16=True, while resume to a fp16=True training, "
                           f"the training process may be unstable.")

        # 4. 恢复 sampler 的状态；
        dataloader_args = self.get_dataloader_args(dataloader)
        if isinstance(dataloader_args.batch_sampler, ReproducibleBatchSampler):
            sampler = dataloader_args.batch_sampler
        elif isinstance(dataloader_args.sampler, ReproducibleSampler):
            sampler = dataloader_args.sampler
        elif isinstance(dataloader_args.sampler, PaddleRandomSampler):
            sampler = RandomSampler(dataloader_args.sampler.data_source)
            logger.debug("Replace paddle RandomSampler into fastNLP RandomSampler.")
        elif self.is_distributed():
            raise RuntimeError("It is not allowed to use checkpoint retraining when you do not use our or "
                               "`ReproducibleSampler`.")
        else:
            sampler = ReproduceBatchSampler(
                batch_sampler=dataloader_args.batch_sampler if dataloader_args.batch_sampler is not None else dataloader_args.sampler,
                batch_size=dataloader_args.batch_size,
                drop_last=dataloader_args.drop_last
            )
        sampler.load_state_dict(states["sampler_states"])
        states["dataloader"] = self.set_dist_repro_dataloader(dataloader, sampler)

        # 5. 修改 trainer_state.batch_idx_in_epoch
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
        r"""
        返回一个不计算梯度的环境用来对模型进行评测；

        :return: context 上下文对象 `paddle.no_grad`;
        """
        return paddle.no_grad

    @staticmethod
    def move_model_to_device(model: "paddle.nn.Layer", device: Union[str, int, "paddle.CUDAPlace", "paddle.CPUPlace"]):
        r"""
        用来将模型转移到指定的 device 上；
        在 Paddle 中使用可能会引起因与设置的设备不一致而产生的问题，请注意。
        """
        if device is not None:
            model.to(device)

    def move_data_to_device(self, batch: "paddle.Tensor"):
        r"""
        将数据迁移到指定的机器上；batch 可能是 list 也可能 dict ，或其嵌套结构。
        在 Paddle 中使用可能会引起因与设置的设备不一致而产生的问题，请注意。

        :return: 将移动到指定机器上的 batch 对象返回；
        """
        device = get_device_from_visible(self.data_device)
        return paddle_move_data_to_device(batch, device)

    @staticmethod
    def worker_init_function(worker_id: int, rank: Optional[int] = None) -> None:  # pragma: no cover
        """The worker_init_fn that Lightning automatically adds to your dataloader if you previously set set the seed
        with ``seed_everything(seed, workers=True)``.

        See also the PyTorch documentation on
        `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
        """
        # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
        global_rank = rank if rank is not None else int(os.environ.get(FASTNLP_GLOBAL_RANK, 0))
        # TODO gpu
        process_seed = paddle.fluid.core.default_cpu_generator().initial_seed()
        # back out the base seed so we can use all the bits
        base_seed = process_seed - worker_id
        ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
        # use 128 bits (4 x 32-bit words)
        np.random.seed(ss.generate_state(4))
        # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
        paddle_ss, stdlib_ss = ss.spawn(2)
        paddle.seed(paddle_ss.generate_state(1, dtype=np.uint64)[0])
        # use 128 bits expressed as an integer
        stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
        random.seed(stdlib_seed)

    def set_deterministic_dataloader(self, dataloader):
        r"""
        为了确定性训练要对 dataloader 进行修改，保证在确定随机数种子后，每次重新训练得到的结果是一样的；
        作用是替换 datalaoder 的 `worker_init_fn`。
        """
        if int(os.environ.get(FASTNLP_SEED_WORKERS, 0)) and dataloader.worker_init_fn is None:
            dataloader.worker_init_fn = partial(self.worker_init_function, rank=self.global_rank)

    def set_sampler_epoch(self, dataloader: "DataLoader", cur_epoch_idx):
        r"""
        对于分布式的 sampler，dataloader 需要在每一个 epoch 前设置随机数种子，来保证每一个进程上的 shuffle 是一样的；

        :param cur_epoch_idx: 当前是第几个 epoch；
        """
        if callable(getattr(dataloader.batch_sampler, "set_epoch", None)):
            dataloader.batch_sampler.set_epoch(cur_epoch_idx)

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

        # paddle 的 DataLoader 一定会有 dataset 属性；
        res.dataset = dataloader.dataset

        if dataloader.batch_sampler is not None:
            # 不过在 paddle 中，我们限定了 batch_sampler 不能为 None
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
                elif isinstance(dataloader.batch_sampler.sampler, PaddleRandomSampler):
                    res.shuffle = True
                else:
                    res.shuffle = False
            # ReproduceBatchSampler 的情况
            elif hasattr(dataloader.batch_sampler, "batch_sampler"):
                batch_sampler = dataloader.batch_sampler.batch_sampler
                res.sampler = batch_sampler.sampler
                if hasattr(batch_sampler.sampler, "shuffle"):
                    res.shuffle = dataloader.batch_sampler.sampler.shuffle
                elif isinstance(batch_sampler.sampler, PaddleRandomSampler):
                    res.shuffle = True
                else:
                    res.shuffle = False
            else:
                res.sampler = None
                res.shuffle = False

            if hasattr(dataloader.batch_sampler, "drop_last"):
                res.drop_last = getattr(dataloader.batch_sampler, "drop_last")
            # 用户使用的是自己的 batch_sampler 并且其没有 "drop_last" 属性；
            else:
                res.drop_last = False

        return res
