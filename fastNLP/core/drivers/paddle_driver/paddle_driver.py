import os
import random
from typing import Union, Optional, Callable, Dict
from functools import partial

import numpy as np

from .utils import _build_fp16_env
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.core.drivers.driver import Driver
from fastNLP.core.utils import apply_to_collection, paddle_move_data_to_device
from fastNLP.envs import rank_zero_call
from fastNLP.envs import FASTNLP_SEED_WORKERS
from fastNLP.core.log import logger

if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle.io import DataLoader, IterableDataset
    from paddle.optimizer import Optimizer

    _reduces = {
        'max': paddle.max,
        'min': paddle.min,
        'mean': paddle.mean,
        'sum': paddle.sum
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

    def zero_grad(self, set_to_none: bool = False):
        r"""
        实现深度学习中的梯度的置零操作，应当直接通过优化器 optimizers 来将梯度置零；
        注意梯度累积不需要在这里实现，trainer 已经在内部实现了梯度累积；

        :param set_to_none: 用来判断是否需要将梯度直接置为 None；Paddle中这个参数无效。
        """
        # if set_to_none:
        #     log.warning("Parameter `set_to_none` does nothing in paddle since grad cannot be set directly.")
        for optimizer in self.optimizers:
            optimizer.clear_grad()

    @staticmethod
    def _check_dataloader_legality(dataloader, dataloader_name, is_train: bool = False):
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

    def check_evaluator_mode(self, mode: str):
        r"""
        因为我们在具体的 driver 的 validate_step 和 test_step 的逻辑是如果模型没有实现本函数，那么就去检测模型是否实现了另一个函数；
        因此如果用户的 evaluator mode 是 validate，但是传入的 model 却没有实现 validate_step 函数，而是实现了 test_step 函数，那么
         我们应当提醒用户这一行为；
        """
        model = self.unwrap_model()
        if mode == "validate":
            if not hasattr(model, "validate_step"):
                if hasattr(model, "test_step"):
                    logger.warning(
                        "Your model does not have 'validate_step' method but has 'test_step' method, but you"
                        "are using 'Evaluator.validate', we are going to use 'test_step' to substitute for"
                        "'validate_step'.")

        else:
            if not hasattr(model, "test_step"):
                if hasattr(model, "validate_step"):
                    logger.warning("Your model does not have 'test_step' method but has 'validate' method, but you"
                                    "are using 'Evaluator.test', we are going to use 'validate_step' to substitute for"
                                    "'test_step'.")

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
    def save_model(self, filepath: str, only_state_dict: bool = True, model_save_fn: Optional[Callable]=None, **kwargs):
        r"""
        保存模型的函数；注意函数 `save` 是用来进行断点重训的函数；
        如果 `model_save_fn` 是一个可调用的函数，那么我们会直接运行该函数；

        :param filepath: 保存文件的文件位置（需要包括文件名）；
        :param only_state_dict: 是否只保存模型的 `state_dict`；注意该参数仅当 `model_save_fn` 为 None 时有效；
        :param model_save_fn: 用户传入的用来代替该函数本身保存逻辑的函数；如果该参数不为 None，那么我们会调用 model_save_fn(path)；
        """
        if model_save_fn is not None:
            model_save_fn(filepath)
        else:
            model = self.unwrap_model()
            if only_state_dict:
                paddle.save(model.state_dict(), filepath)
            else:
                input_spec = kwargs.get("input_spec", None)
                if input_spec is None:
                    raise Exception("To save the whole Paddle Layer, parameter 'input_spec' is needed.")
                paddle.jit.save(model, filepath, input_spec)

    @staticmethod
    @rank_zero_call
    def load_model(filepath: str, load_dict: bool = True):
        r"""
        加载模型的函数；注意函数 `load` 是用来进行断点重训的函数；

        :param filepath: 需要被加载的对象的文件位置（需要包括文件名）；
        :param load_dict: 是否加载state_dict，默认为True。当用户在save_model时将only_state_dict设置为False时，
                          即保存了整个模型时，这个参数必须也为False
        :return: 返回加载指定文件后的结果；
        """
        if load_dict:
            return paddle.load(filepath)
        else:
            return paddle.jit.load(filepath)

    @rank_zero_call
    def save(self, folder, states: Dict):
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
        """
        # 1. 保存模型的状态；
        model = self.unwrap_model()
        model_state_dict = {name: param.cpu().detach().clone() for name, param in model.state_dict().items()}
        # 对于单卡的 driver 来讲，我们实际上（现在）不应该考虑用户在DDP环境下使用单卡模式，从而造成效率损失；
        states["model_state_dict"] = model_state_dict

        # 2. 保存 optimizers 的状态；
        optimizers_state_dict = {}
        for i in range(len(self.optimizers)):
            optimizer: Optimizer = self.optimizers[i]
            optimizer_state = optimizer.state_dict()
            optimizer_state = {name: param.cpu().detach().clone() for name, param in optimizer_state.items()}
            optimizers_state_dict[f"optimizer{i}"] = optimizer_state  # 注意这里没有使用 deepcopy，测试是不需要的；
        states["optimizers_state_dict"] = optimizers_state_dict

        paddle.save(states, folder)

    def load(self, filepath) -> Dict:
        r"""
        断点重训的加载函数，注意该函数会负责读取数据，并且恢复模型和 optimizers 的 state_dict 等；
            driver 实例需要在该函数中先加载模型和 optimizers 的 state_dict，然后将一个 state 字典返回给 trainer 。
            因此 save 函数和 load 函数的接受和返回值应该是对应的；

        该函数需要在所有 rank 上执行。

        :param filepath: 保存断点重训的状态的文件名；
        :return: 需要返回 save 函数输入的 states 内容；
        """
        states = paddle.load(filepath)

        # 1. 加载 optimizers 的状态；
        optimizers_state_dict = states["optimizers_state_dict"]
        for i in range(len(self.optimizers)):
            optimizer: paddle.optimizer.Optimizer = self.optimizers[i]
            optimizer.set_state_dict(optimizers_state_dict[f"optimizer{i}"])

        # 2. 加载模型状态；
        model = self.unwrap_model()
        model.load_dict(states["model_state_dict"])

        self.barrier()
        return states

    def get_evaluate_context(self):
        r"""
        返回一个不计算梯度的环境用来对模型进行评测；

        :return: context 上下文对象 `paddle.no_grad`;
        """
        return paddle.no_grad

    @staticmethod
    def move_model_to_device(model: 'paddle.nn.Layer', device: Union[str, int, 'paddle.CUDAPlace', 'paddle.CPUPlace']):
        r"""
        用来将模型转移到指定的 device 上；
        在 Paddle 中使用可能会引起因与设置的设备不一致而产生的问题，请注意。
        """
        if device is not None:
            model.to(device)

    def move_data_to_device(self, batch: 'paddle.Tensor'):
        r"""
        将数据迁移到指定的机器上；batch 可能是 list 也可能 dict ，或其嵌套结构。
        在 Paddle 中使用可能会引起因与设置的设备不一致而产生的问题，请注意。

        :return: 将移动到指定机器上的 batch 对象返回；
        """
        return paddle_move_data_to_device(batch, self.data_device)

    @staticmethod
    def worker_init_function(worker_id: int, rank: Optional[int] = None) -> None:  # pragma: no cover
        """The worker_init_fn that Lightning automatically adds to your dataloader if you previously set set the seed
        with ``seed_everything(seed, workers=True)``.

        See also the PyTorch documentation on
        `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
        """
        # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
        global_rank = rank if rank is not None else rank_zero_call.rank
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

    def set_sampler_epoch(self, dataloader: 'DataLoader', cur_epoch_idx):
        r"""
        对于分布式的 sampler，dataloader 需要在每一个 epoch 前设置随机数种子，来保证每一个进程上的 shuffle 是一样的；

        :param cur_epoch_idx: 当前是第几个 epoch；
        """
        if callable(getattr(dataloader.batch_sampler, "set_epoch", None)):
            dataloader.batch_sampler.set_epoch(cur_epoch_idx)
