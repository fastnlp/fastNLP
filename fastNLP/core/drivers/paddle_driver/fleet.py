r"""
用于实现 **PaddlePaddle** 框架下使用 ``fleet`` 分布式训练 API 进行集群式（*collective*）多卡训练的 Driver。

.. note::

    在 **PaddlePaddle** 框架中，使用分布式训练的方式可以参见 **PaddlePaddle** 的
    `官方文档 <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/cluster_quick_start_cn.html>`_ 。
    简言之，分布式训练的过程可以概括为：导入 ``fleet`` 包 -> 使用 :func:`fleet.init` 初始化分布式环境 -> 初始化模型，转换为并行模型开始训练。

**fastNLP** 支持三种启动分布式训练的方式（假设执行训练的文件名为 ``train.py``）：

    A. 用户自己不进行分布式的任何操作，直接使用我们的 :class:`~fastNLP.core.Trainer` 进行训练，此时将参数 ``device``
       设置为一个列表，然后使用 ``python train.py`` 的方式开始训练；
    B. 用户自己不进行分布式的任何操作，但是使用 ``python -m paddle.distributed.launch train.py`` 开始训练；
    C. 用户自己在外面初始化分布式环境，并且通过 ``python -m paddle.distributed.launch train.py`` 开始训练；

.. note::

    在后两种启动方式中，您需要通过参数 ``--gpus`` 来指定训练使用的设备，在 ``trainer`` 中设置的参数是无效的。

不过在使用该 Driver 之前，我们需要向您说明 **fastNLP** 实现 ``PaddleFleetDriver`` 的思路，以便于您理解代码编写过程中可能出现的问题。

在 **fastNLP** 中，为了尽可能减少单卡向分布式训练转换过程中的代码变动，我们需要在 ``PaddleFleetDriver`` 中进行 **分布式环境初始化**
和 **将模型转换为并行模式** 等操作，同时实现多卡训练的方法是从主进程（``rank=0``）中创建其它的所有子进程（``rank=1,2,...``）。
在这个过程中，我们发现由于 **PaddlePaddle** 框架的特性，会出现下面的问题：

    1. **fastNLP** 中，初始化模型一定会在初始化 ``Driver`` 之前，因此调用 :func:`fleet.init` 的时机会在初始化模型之后；
       此时子进程中模型将无法正常地初始化，提示无法找到设备 ``gpu:0``；
    2. 在训练的过程中，会出现训练一个 ``batch`` 后程序卡住或程序会占用所有可见显卡的情况；

考虑到这些问题，我们为 **PaddlePaddle** 的分布式训练制定了这样的约束：在导入 **fastNLP** 之前，必须设置环境变量 ``FASTNLP_BACKEND``
为 ``paddle``。执行方法有两种::

    >>> import os
    >>> os.environ["FASTNLP_BACKEND"] = "paddle" # 设置环境变量
    >>> import fastNLP # 设置之后才可以导入 fastNLP

或是在执行脚本（假设文件名为 ``train.py`` ）时设置::

    FASTNLP_BACKEND=paddle python train.py
    FASTNLP_BACKEND=paddle python -m paddle.distributed.lauch train.py

设置 ``FASTNLP_BACKEND=paddle`` 后，**fastNLP** 会在 ``import paddle`` 之前通过 ``CUDA_VISIBLE_DEVICES`` 将设备限制在所有可见设备的第
**0** 张卡上，以此绕开通信和同步上的种种限制。我们会将用户希望可见的设备（如用户自己设置了 ``CUDA_VISIBLE_DEVICES`` 的情况）保存在另一个环境变量
``USER_CUDA_VISIBLE_DEVICES`` 中来确保 **fastNLP** 能够知道用户的设置。假设用户希望在 ``[0,2,3]`` 三张显卡上进行分布式训练，那么在三个训练进程中，
``CUDA_VISIBLE_DEVICES`` 就分别为 0、2 和 3 。

.. note::

    我们会事先将设备限制在所有可见设备的第 **0** 张卡上，因此多卡训练的参数 ``device`` 一定要以 **0** 开始，否则会无法正常地启动。
    如果您希望调整使用的第一张显卡，请使用 ``CUDA_VISIBLE_DEVICES`` 进行限制。

.. note::

    根据 **PaddlePaddle** 的说明，设置 ``CUDA_VISIBLE_DEVICES`` 之后启动分布式训练时，情况A与情况BC设置设备的方式会有所不同。
    情况A应设置为实际设备相对可见设备的索引，而情况BC应设置为实际的设备号：

        1. 情况A中， ``CUDA_VISIBLE_DEVICES=3,4,5,6`` 且参数 ``device=[0,2,3]`` 代表使用 **3号、5号和6号** 显卡；
        2. 情况BC中，``CUDA_VISIBLE_DEVICES=3,4,5,6`` 且参数 ``--gpu=3,5,6`` 代表使用 **3号、5号和6号** 显卡；

.. note::

    多机的启动强制要求用户在每一台机器上使用 ``python -m paddle.distributed.launch`` 启动；因此我们不会在 ``PaddleFleetDriver``
    中保存任何当前有多少台机器的信息。

"""
import os
from typing import List, Union, Optional, Dict, Tuple, Callable

from .paddle_driver import PaddleDriver
from .fleet_launcher import FleetLauncher
from .utils import (
    _FleetWrappingModel, 
    replace_sampler,
    replace_batch_sampler,
    _check_dataloader_args_for_distributed
)
from .dist_utils import fastnlp_paddle_all_gather, fastnlp_paddle_broadcast_object

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.core.utils import (
    auto_param_call,
    check_user_specific_params,
    is_in_paddle_dist,
    get_paddle_device_id,
)
from fastNLP.core.utils.paddle_utils import _convert_data_device
from fastNLP.envs.distributed import rank_zero_rm
from fastNLP.core.samplers import (
    ReproduceBatchSampler,
    ReproducibleSampler,
    ReproducibleBatchSampler,
    RandomSampler,
    UnrepeatedSampler,
    UnrepeatedSequentialSampler,
    re_instantiate_sampler,
    conversion_between_reproducible_and_unrepeated_sampler,
)
from fastNLP.envs.env import (
    FASTNLP_DISTRIBUTED_CHECK,
    FASTNLP_GLOBAL_SEED,
    FASTNLP_NO_SYNC,
    USER_CUDA_VISIBLE_DEVICES,
)
from fastNLP.core.log import logger

if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle import DataParallel
    import paddle.distributed.fleet as fleet
    import paddle.distributed as paddledist
    from paddle.optimizer import Optimizer
    from paddle.fluid.reader import _DatasetKind
    from paddle.fluid.dygraph import parallel_helper
    from paddle.io import BatchSampler

__all__ = [
    "PaddleFleetDriver",
]

class PaddleFleetDriver(PaddleDriver):
    """
    :param model: 训练使用的模型。

        * 如果不想自己初始化分布式环境，类型应为 :class:`paddle.nn.Layer`；
        * 如果已经在外面初始化了分布式环境，类型应为 :class:`paddle.DataParallel`；

    :param parallel_device: 多卡训练时使用的设备，必须是一个列表。
        当使用 ``python -m paddle.distributed.launch`` 启动时，该参数无效。
    :param is_pull_by_paddle_run: 标记当前进程是否为通过 ``python -m paddle.distributed.launch`` 启动的。
        这个参数仅在 :class:`~fastNLP.core.Trainer` 中初始化 driver 时使用
    :param fp16: 是否开启混合精度训练
    :param paddle_kwargs:
        * *fleet_kwargs* -- 用于在使用 ``PaddleFleetDriver`` 时指定 ``DataParallel`` 和 ``fleet`` 初始化时的参数，包括：
            
            * *is_collective* -- 是否使用 paddle 集群式的分布式训练方法，目前仅支持为 ``True`` 的情况。
            * *role_maker* -- 初始化 ``fleet`` 分布式训练 API 时使用的 ``RoleMaker``。
            * 其它用于初始化 ``DataParallel`` 的参数。
        * *gradscaler_kwargs* -- 用于 ``fp16=True`` 时，提供给 :class:`paddle.amp.GradScaler` 的参数
    
    :kwargs:
        * *wo_auto_param_call* (``bool``) -- 是否关闭在训练时调用我们的 ``auto_param_call`` 函数来自动匹配 batch 和前向函数的参数的行为

        .. note::

            关于该参数的详细说明，请参见 :class:`~fastNLP.core.controllers.Trainer` 中的描述；函数 ``auto_param_call`` 详见 :func:`fastNLP.core.utils.auto_param_call`。

    """
    def __init__(
            self, 
            model, 
            parallel_device: Optional[Union[List[str], str]],
            is_pull_by_paddle_run: bool = False,
            fp16: bool = False,
            paddle_kwargs: Dict = None,
            **kwargs
    ):
        if USER_CUDA_VISIBLE_DEVICES not in os.environ:
            raise RuntimeError("To run paddle distributed training, please set `FASTNLP_BACKEND` to 'paddle' before using fastNLP.")
        super(PaddleFleetDriver, self).__init__(model, fp16=fp16, paddle_kwargs=paddle_kwargs, **kwargs)

        # 如果不是通过 launch 启动，要求用户必须传入 parallel_device
        if not is_pull_by_paddle_run:
            if parallel_device is None:
                raise ValueError("Parameter `parallel_device` can not be None when using `PaddleFleetDriver`. This error is caused "
                                "when your value of parameter `device` is `None` in your `Trainer` instance.")
            if not isinstance(parallel_device, List):
                raise ValueError("Parameter `parallel_device`'s type must be List when using `PaddleFleetDriver`, "
                                    f"not {type(parallel_device)}.")
            if get_paddle_device_id(parallel_device[0]) != 0:
                raise ValueError("The first device of `parallel_device` must be 'gpu:0' in fastNLP.")
        
        # 如果用户自己初始化了 paddle 的分布式训练那么一定是通过 launch 拉起的
        # 这个参数会在 initialize_paddle_drvier 中设置。
        self.is_pull_by_paddle_run = is_pull_by_paddle_run
        self.parallel_device = parallel_device
        # 在初始化时，如果发现 is_pull_by_paddle_run ，则将 parallel_device 设置成当前进程的gpu
        if is_pull_by_paddle_run:
            self.model_device = parallel_device
        else:
            self.model_device = parallel_device[self.local_rank]

        # 如果用户自己在外面初始化了并行模型；
        self.outside_fleet = False
        if parallel_helper._is_parallel_ctx_initialized() and FASTNLP_DISTRIBUTED_CHECK not in os.environ and \
                "fastnlp_paddle_launch_not_fleet" not in os.environ:
            # 如果用户自己在外面初始化了 Fleet，那么我们要求用户传入的模型一定是已经由 DistributedDataParallel 包裹后的模型；
            if not isinstance(model, DataParallel):
                raise RuntimeError(
                    "It is not allowed to input a normal model instead of `paddle.DataParallel` when"
                    "you initialize the paddle distribued process out of our control.")

            self.outside_fleet = True

        self.world_size = None
        self.global_rank = 0
        self.gloo_rendezvous_dir = None
        
        self._fleet_kwargs = self._paddle_kwargs.get("fleet_kwargs", {})
        check_user_specific_params(self._fleet_kwargs, DataParallel.__init__, DataParallel.__name__)
        # fleet.init 中对于分布式策略的设置，详情可以参考 PaddlePaddle 的官方文档
        self.strategy = self._fleet_kwargs.get("strategy", fleet.DistributedStrategy())
        self.is_collective = self._fleet_kwargs.pop("is_collective", True)
        if not self.is_collective:
            raise NotImplementedError("fastNLP only support `collective` for distributed training now.")
        self.role_maker = self._fleet_kwargs.pop("role_maker", None)

        self.output_from_new_proc = kwargs.get("output_from_new_proc", "only_error")
        assert isinstance(self.output_from_new_proc, str), "Parameter `output_from_new_proc` can only be `str` type."
        if self.output_from_new_proc not in {"all", "ignore", "only_error"}:
            os.makedirs(name=self.output_from_new_proc, exist_ok=True)
            self.output_from_new_proc = os.path.abspath(self.output_from_new_proc)

        self._has_setup = False # 设置这一参数是因为 evaluator 中也会进行 setup 操作，但是显然是不需要的也不应该的；
        self._has_fleetwrapped = False  # 判断传入的模型是否经过 _has_fleetwrapped 包裹；

    def setup(self):
        """
        初始化分布式训练的环境。

        1. 如果是通过 ``paddle.distributed.launch`` 方法启动的，则根据已经设置好的环境获取分布式的属性。
        2. 否则启动子进程。
        """
        if self._has_setup:
            return
        self._has_setup = True
        # 如果用户需要使用多机模式，那么一定进入到这里；
        if self.is_pull_by_paddle_run:

            if self.outside_fleet:
                # 已经初始化了多机环境
                self._set_from_fleet_environment()
            else:
                # 用户没有初始化多机环境
                # TODO 绕一下
                # dist.get_world_size() 只能在初始化之后进行调用；
                self.world_size = int(os.environ.get("PADDLE_TRAINERS_NUM"))
                self.global_rank = int(os.environ.get("PADDLE_TRAINER_ID"))
                logger.info(f"World size: {self.world_size}, Global rank: {self.global_rank}")
                if not parallel_helper._is_parallel_ctx_initialized():
                    fleet.init(self.role_maker, self.is_collective, self.strategy)

                os.environ["fastnlp_paddle_launch_not_fleet"] = "yes"

        else:
            # 在用户只使用了一个分布式 trainer 的情况下
            # 此时 parallel_helper._is_parallel_ctx_initialized() 一定为 False
            # parallel_device 是 list，
            if not parallel_helper._is_parallel_ctx_initialized():
                # 拉起子进程并设置相应的属性
                self._init_fleet_and_set()
            # 用户在这个 trainer 前面又初始化了一个 trainer，并且使用的是 PaddleFleetDriver；
            else:
                # 已经设置过一次，保证参数必须是一样的
                pre_gpus = os.environ[FASTNLP_DISTRIBUTED_CHECK]
                pre_gpus = [int(x) for x in pre_gpus.split(",")]
                cur_gpus = [get_paddle_device_id(g) for g in self.parallel_device]
                if sorted(pre_gpus) != sorted(self.parallel_device):
                    raise RuntimeError("Notice you are using `PaddleFleetDriver` after one instantiated `PaddleFleetDriver`, it is not"
                                    "allowed that your second `PaddleFleetDriver` has a new setting of parameters `parallel_device`.")
                self.world_size = paddledist.get_world_size()
                self.global_rank = paddledist.get_rank()

        if not self.outside_fleet:
            # self.model.to(self.model_device)
            self.configure_fleet()

        self.barrier()

        # 初始化 self._pids，从而使得每一个进程都能接受到 rank0 的 send 操作；
        # TODO 不用.to会怎么样？
        self._pids = []
        paddledist.all_gather(self._pids, paddle.to_tensor(os.getpid(), dtype="int32"))
        # TODO LOCAL_WORLD_SIZE
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE")) if "LOCAL_WORLD_SIZE" in os.environ else None
        if local_world_size is None:
            local_world_size = paddle.to_tensor(self.local_rank, dtype="int32")
            paddledist.all_reduce(local_world_size, op=paddledist.ReduceOp.MAX)
            local_world_size = local_world_size.item() + 1

        node_rank = self.global_rank // local_world_size
        self._pids = self._pids[node_rank*local_world_size: (node_rank+1)*local_world_size]
        self._pids = self.tensor_to_numeric(self._pids)

    def _init_fleet_and_set(self):
        """
        使用 FleetLauncher 拉起子进程
        """
        if self.local_rank == 0:
            logger._set_distributed()
            # 是 rank0 的话，则拉起其它子进程
            launcher = FleetLauncher(self.parallel_device, self.output_from_new_proc)
            launcher.launch()
            self.gloo_rendezvous_dir = launcher.gloo_rendezvous_dir
        # 设置参数和初始化分布式环境
        fleet.init(self.role_maker, self.is_collective, self.strategy)
        self.global_rank = int(os.getenv("PADDLE_TRAINER_ID"))
        self.world_size = int(os.getenv("PADDLE_TRAINERS_NUM"))

        # 正常情况下不会 Assert 出问题，但还是保险一下
        assert self.global_rank is not None
        assert self.world_size is not None
        assert self.world_size == len(self.parallel_device)

    def _set_from_fleet_environment(self):
        """
        当用户使用了 `python -m paddle.distributed.launch xxx.py` 启动时，我们需要
        根据 paddle 设置的环境变量来获得各种属性
        """
        self.world_size = paddledist.get_world_size()
        self.global_rank = paddledist.get_rank()

    def barrier(self):
        """
        同步进程之间的操作
        """
        if int(os.environ.get(FASTNLP_NO_SYNC, 0)) < 1:  # 当 FASTNLP_NO_SYNC 小于 1 时实际执行
            paddledist.barrier()

    def configure_fleet(self):
        # 将模型用 DataParallel 和自定义的类型包裹起来
        if not self._has_fleetwrapped and not isinstance(self.model, DataParallel):
            self.model = DataParallel(
                _FleetWrappingModel(self.model),
                **self._fleet_kwargs
            )
            self._has_fleetwrapped = True

    def on_exception(self):
        rank_zero_rm(self.gloo_rendezvous_dir)
        super().on_exception()

    @property
    def world_size(self) -> int:
        """
        分布式训练的进程总数 ``WOLRD_SIZE``
        """
        return self._world_size

    @world_size.setter
    def world_size(self, size: int) -> None:
        self._world_size = size

    @property
    def global_rank(self) -> int:
        """
        当前进程的全局编号 ``global_rank``
        """
        return self._global_rank

    @global_rank.setter
    def global_rank(self, rank: int) -> None:
        self._global_rank = rank

    @property
    def local_rank(self) -> int:
        """
        当前进程的局部编号 ``local_rank``
        """
        return int(os.getenv("PADDLE_RANK_IN_NODE", "0"))

    @property
    def data_device(self):
        """
        数据所在的设备；由于 **PaddlePaddle** 可以通过环境变量获取当前进程的设备，因此该属性
        和 ``model_device`` 表现相同。
        """
        return self.model_device

    def model_call(self, batch, fn: Callable, signature_fn: Optional[Callable]) -> Dict:
        if self._has_fleetwrapped:
            return self.model(batch, fastnlp_fn=fn, fastnlp_signature_fn=signature_fn,
                              wo_auto_param_call=self.wo_auto_param_call)
        else:
            if isinstance(batch, Dict) and not self.wo_auto_param_call:
                return auto_param_call(fn, batch, signature_fn=signature_fn)
            else:
                return fn(batch)

    def get_model_call_fn(self, fn: str) -> Tuple:
        model = self.unwrap_model()
        if self._has_fleetwrapped:
            if hasattr(model, fn):
                fn = getattr(model, fn)
                if not callable(fn):
                    raise RuntimeError(f"The `{fn}` attribute of model is not `Callable`.")
                return fn, None
            elif fn in {"train_step", "evaluate_step"}:
                return model, model.forward
            else:
                raise RuntimeError(f"There is no `{fn}` method in your model.")
        else:
            if hasattr(model, fn):
                logger.warning("Notice your model is a `DataParallel` model. And your model also implements "
                               f"the `{fn}` method, which we can not call actually, we will"
                               " call `forward` function instead of `train_step` and you should note that.")
            elif fn not in {"train_step", "evaluate_step"}:
                raise RuntimeError(f"There is no `{fn}` method in your model. And also notice that your model is a "
                                   "`DistributedDataParallel` model, which means that we will only call model.forward "
                                   "function when we are in forward propagation.")

            return self.model, model.forward

    def set_dist_repro_dataloader(self, dataloader, dist: Optional[Union[str, ReproducibleSampler, ReproduceBatchSampler]],
                                  reproducible: bool = False):
        # 暂时不支持iterableDataset
        assert dataloader.dataset_kind != _DatasetKind.ITER, \
                    "FastNLP does not support `IteratorDataset` now."
        # 如果 dist 为 ReproducibleBatchSampler, ReproducibleSampler 说明是在断点重训时 driver.load_checkpoint 函数调用；
        if isinstance(dist, ReproducibleBatchSampler):
            dist.set_distributed(
                num_replicas=self.world_size,
                rank=self.global_rank,
                pad=True
            )
            return replace_batch_sampler(dataloader, dist)
        if isinstance(dist, ReproducibleSampler):
            dist.set_distributed(
                num_replicas=self.world_size,
                rank=self.global_rank,
                pad=True
            )
            return replace_sampler(dataloader, dist)

       # 如果 dist 为 str 或者 None，说明是在 trainer 初试化时调用；
        # trainer, evaluator
        if dist is None:
            if reproducible:
                raise RuntimeError("It is not allowed to save checkpoint if the sampler is not allowed to be replaced.")
            else:
                args = self.get_dataloader_args(dataloader)
                if isinstance(args.batch_sampler, ReproducibleBatchSampler):
                    batch_sampler = re_instantiate_sampler(args.batch_sampler)
                    return replace_batch_sampler(dataloader, batch_sampler)
                if isinstance(args.sampler, ReproducibleSampler):
                    sampler = re_instantiate_sampler(args.sampler)
                    return replace_sampler(dataloader, sampler)
                return dataloader
        # trainer
        elif dist == "dist":
            args = self.get_dataloader_args(dataloader)
            # 如果用户的 trainer.use_dist_sampler 为 True，那么此时其是否进行断点重训，不影响这里的行为；
            if isinstance(args.batch_sampler, ReproducibleBatchSampler):
                batch_sampler = re_instantiate_sampler(args.batch_sampler)
                batch_sampler.set_distributed(
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                    pad=True
                )
                return replace_batch_sampler(dataloader, batch_sampler)
            elif isinstance(args.sampler, ReproducibleSampler):
                sampler = re_instantiate_sampler(args.sampler)
                sampler.set_distributed(
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                    pad=True
                )
                return replace_sampler(dataloader, sampler)
            else:
                _check_dataloader_args_for_distributed(args, controller='Trainer')
                sampler = RandomSampler(
                    dataset=args.dataset,
                    shuffle=args.shuffle,
                    seed=int(os.environ.get(FASTNLP_GLOBAL_SEED, 0))
                )
                sampler.set_distributed(
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                    pad=True
                )
                return replace_sampler(dataloader, sampler)
        # evaluator
        elif dist == "unrepeatdist":
            args = self.get_dataloader_args(dataloader)
            if type(args.batch_sampler) != BatchSampler:
                # TODO 这里的目的是判断用户的 batch_sampler 是定制的，可能需要完善
                logger.warning("Note that you are using customized ``batch_sampler`` in evaluate dataloader or" \
                                "train dataloader while testing ``overfit_batches``, which may cause that" \
                                "the data for distributed evaluation is not unrepeated.")
            if isinstance(args.sampler, ReproducibleSampler):
                sampler = conversion_between_reproducible_and_unrepeated_sampler(args.sampler)
            elif not isinstance(args.sampler, UnrepeatedSampler):
                _check_dataloader_args_for_distributed(args, controller='Evaluator')
                sampler = UnrepeatedSequentialSampler(
                    dataset=args.dataset
                )
            else:
                sampler = re_instantiate_sampler(args.sampler)
            sampler.set_distributed(
                num_replicas=self.world_size,
                rank=self.global_rank
            )
            # TODO 这里暂时统一替换为 BatchSampler
            batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=False)
            return replace_batch_sampler(dataloader, batch_sampler)
        else:
            raise ValueError("Parameter `dist_sampler` can only be one of three values: ('dist', 'unrepeatdist', None).")

    def is_global_zero(self) -> bool:
        r"""
        :return: 当前的进程是否在全局上是进程 0
        """
        return self.global_rank == 0

    def get_model_no_sync_context(self):
        r"""
        :return: 一个 ``context`` 上下文环境，用于关闭各个进程之间的同步。
        """
        return self.model.no_sync

    def unwrap_model(self) -> "paddle.nn.Layer":
        """
        获得 driver 最原始的模型。该函数可以取出被 :class:`paddle.DataParallel` 包裹起来的模型。
        """
        _layers = self.model._layers
        if isinstance(_layers, _FleetWrappingModel):
            return _layers.model
        else:
            return _layers

    def get_local_rank(self) -> int:
        r"""
        :return: 当前进程局部的进程编号。
        """
        return self.local_rank

    def is_distributed(self) -> bool:
        """
        :return: 当前使用的 driver 是否是分布式的 driver，在 ``PaddleFleetDriver`` 中，返回 ``True``。
        """
        return True

    @staticmethod
    def _check_optimizer_legality(optimizers):
        # paddle 存在设置分布式 optimizers 的函数，返回值为 fleet.meta_optimizers.HybridParallelOptimizer
        DistribuedOptimizer = fleet.meta_optimizers.HybridParallelOptimizer
        for each_optimizer in optimizers:
            if not isinstance(each_optimizer, (Optimizer, DistribuedOptimizer)):
                raise ValueError(f"Each optimizer of parameter `optimizers` should be 'paddle.optimizer.Optimizer' type, "
                                f"not {type(each_optimizer)}.")

    def broadcast_object(self, obj, src:int=0, group=None, **kwargs):
        r"""
        从 ``src`` 端将 ``obj`` 对象（可能是 tensor ，可能是 object ）广播到其它进程。如果是非 tensor 的对象会尝试使用 pickle 进行打包进行
        传输，然后在接收处处再加载回来。仅在分布式的 driver 中有实际意义。

        :param obj: obj，可能是 Tensor 或 嵌套类型的数据
        :param src: 发送方的 ``global_rank``
        :param group: 进程所在的通信组
        :return: 如果当前 rank 是接收端，则返回接收到的参数；如果是 source 端则返回发送的内容。如果环境变量 ``FASTNLP_NO_SYNC`` 为 **2** 则
            返回 ``None``
        """
        # 因为设置了CUDA_VISIBLE_DEVICES，可能会引起错误
        device = _convert_data_device(self.data_device)
        return fastnlp_paddle_broadcast_object(obj, src, device=device, group=group)

    def all_gather(self, obj, group=None) -> List:
        r"""
        将 ``obj`` 互相传送到其它所有的 rank 上，其中 ``obj`` 可能是 Tensor，也可能是嵌套结构的 object 。如果不是基础类型的数据，将会尝试通过
        pickle 进行序列化，接收到之后再反序列化。

        example::

            >>> # rank 0
            >>> obj = {'a': 1, 'b':[1, 2], 'c':{'d': 1}}
            >>> # rank 1
            >>> obj = {'a': 1, 'b':[1, 2], 'c':{'d': 2}}
            >>> # after all_gather():
            >>> result = [
                    {'a': 1, 'b':[1, 2], 'c':{'d': 1}},
                    {'a': 1, 'b':[1, 2], 'c':{'d': 2}}
                ]

        :param obj: 需要传输的对象，在每个 rank 上都应该保持相同的结构。
        :param group: 进程所在的通信组。
        :return: 所有 rank 发送的 ``obj`` 聚合在一起的内容；如果环境变量 ``FASTNLP_NO_SYNC`` 为 **2** 则不会执行，直接返回 ``[obj]`` 。
        """
        return fastnlp_paddle_all_gather(obj, group=group)
