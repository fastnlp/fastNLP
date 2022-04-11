import os
from functools import partial
from typing import List, Union, Optional, Dict

from .paddle_driver import PaddleDriver
from .fleet_launcher import FleetLauncher
from .utils import (
    _FleetWrappingModel, 
    ForwardState, 
    _MODE_PARAMETER,
    get_device_from_visible,
    reset_seed,
    replace_sampler,
    replace_batch_sampler,
)

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.core.utils import (
    auto_param_call,
    check_user_specific_params,
    paddle_move_data_to_device,
    is_in_paddle_dist,
)
from fastNLP.core.samplers import (
    RandomBatchSampler,
    ReproducibleSampler,
    ReproducibleBatchSampler,
    RandomSampler,
    UnrepeatedSampler,
    UnrepeatedSequentialSampler,
    re_instantiate_sampler,
    conversion_between_reproducible_and_unrepeated_sampler,
)
from fastNLP.envs.env import FASTNLP_DISTRIBUTED_CHECK, FASTNLP_GLOBAL_SEED
from fastNLP.core.log import logger

if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle import DataParallel
    import paddle.distributed.fleet as fleet
    import paddle.distributed as dist
    from paddle.io import BatchSampler
    from paddle.optimizer import Optimizer
    from paddle.fluid.reader import _DatasetKind
    from paddle.fluid.dygraph import parallel_helper

__all__ = [
    "PaddleFleetDriver",
]
# if os.path.exists(self.gloo_rendezvous_dir):
#             shutil.rmtree(self.gloo_rendezvous_dir)
class PaddleFleetDriver(PaddleDriver):
    def __init__(
            self, 
            model, 
            parallel_device: Optional[Union[List[int], int]],
            is_pull_by_paddle_run: bool = False,
            fp16: bool = False,
            **kwargs
    ):
        """
        采用fleet接口进行并行paddle训练的driver
        PaddleFleetDriver 目前考虑支持的三种启动方式：
        1. 用户自己不进行 fleet 的任何操作，直接使用我们的 Trainer，并且只运行一个 main 脚本，这时是由我们自己使用 open_subprocesses 拉起
         多个进程，然后由 Driver 自己进行初始化
        2. 其它情况同 1，但是用户自己使用 python -m paddle.distributed.launch 拉起；
        3. 用户自己在外面初始化 Fleet，并且通过 python -m paddle.distributed.launch 拉起；

        注意多机的启动强制要求用户在每一台机器上使用 python -m paddle.distributed.launch 启动；

        如果用户自己在外面初始化了 fleet，那么
            parallel_device 为 None；
            data_device 为 表示单卡的一个参数；
            dist.is_initialized 为 true；
        """
        super(PaddleFleetDriver, self).__init__(model, fp16=fp16, **kwargs)

        # 如果不是通过 launch 启动，要求用户必须传入 parallel_device
        if not is_pull_by_paddle_run and parallel_device is None:
            raise ValueError("Parameter `parallel_device` can not be None when using `PaddleFleetDriver`. This error is caused "
                             "when your value of parameter `device` is `None` in your `Trainer` instance.")
        
        # 如果用户自己初始化了 paddle 的分布式训练那么一定是通过 launch 拉起的
        self.is_pull_by_paddle_run = is_pull_by_paddle_run
        self.parallel_device = parallel_device
        # 在初始化时，如果发现 is_pull_by_paddle_run ，则将 parallel_device 设置成当前进程的gpu
        if is_pull_by_paddle_run:
            self._model_device = parallel_device
        else:
            self._model_device = parallel_device[self.local_rank]

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
            # 用户只有将模型上传到对应机器上后才能用 DataParallel 包裹，因此如果用户在外面初始化了 Fleet，那么在 PaddleFleetDriver 中
            #  我们就直接将 model_device 置为 None；
            self._model_device = None

            def _running_fn_(batch, step_fn, signature_fn):
                if isinstance(batch, Dict):
                    return auto_param_call(step_fn, batch, signature_fn=signature_fn)
                else:
                    return self._validate_step(batch)

            model = model._layers
            if hasattr(model, "train_step"):
                logger.warning(
                    "Notice your model is a `paddle.DataParallel` model. And your "
                    "model also implements the `train_step` method, which we can not call actually, we will"
                    " call `forward` function instead of `train_step` and you should note that.")
            self._train_step = partial(_running_fn_, step_fn=self.model, signature_fn=model.forward)
            # self._train_signature_fn = model.forward

            if hasattr(model, "validate_step"):
                logger.warning(
                    "Notice your model is a `paddle.DataParallel` model. And your "
                    "model also implements the `validate_step` method, which we can not call actually, "
                    "we will call `forward` function instead of `validate_step` and you should note that.")
            self._validate_step = partial(_running_fn_, step_fn=self.model, signature_fn=model.forward)
            # self._validate_signature_fn = model.forward

            if hasattr(model, "test_step"):
                logger.warning(
                    "Notice your model is a `paddle.DataParallel` model. And your "
                    "model also implements the `test_step` method, which we can not call actually, we will"
                    " call `forward` function instead of `test_step` and you should note that.")
            self._test_step = partial(_running_fn_, step_fn=self.model, signature_fn=model.forward)

        # 当参数 `device` 为 None 时并且该参数不为 None，表示将对应的数据移到指定的机器上；
        self._data_device = kwargs.get("data_device", None)
        if self._data_device is not None:
            if isinstance(self._data_device, int):
                if self._data_device < 0:
                    raise ValueError("Parameter `data_device` can not be smaller than 0.")
                _could_use_device_num = paddle.device.cuda.device_count()
                if self._data_device >= _could_use_device_num:
                    raise ValueError("The gpu device that parameter `device` specifies is not existed.")
                self._data_device = f"gpu:{self._data_device}"
            elif not isinstance(self._data_device, str):
                raise ValueError("Parameter `device` is wrong type, please check our documentation for the right use.")
            if self.outside_fleet and paddle.device.get_device() != self._data_device:
                logger.warning("`Parameter data_device` is not equal to paddle.deivce.get_device(), "
                                "please keep them equal to avoid some potential bugs.")

        self.world_size = None
        self.global_rank = 0
        self._configured = False  # 防止重复调用 configure_ddp() 函数使用
        self._has_setup = False # 防止重复调用 setup() 函数

        self._fleet_kwargs = kwargs.get("paddle_fleet_kwargs", {})
        check_user_specific_params(self._fleet_kwargs, DataParallel.__init__)
        self.strategy = self._fleet_kwargs.get("strategy", fleet.DistributedStrategy())
        self.is_collective = self._fleet_kwargs.get("is_collective", True)
        if not self.is_collective:
            raise NotImplementedError("FastNLP only support `collective` for distributed training now.")
        self.role_maker = self._fleet_kwargs.get("role_maker", None)

        if self.local_rank == 0 and not is_in_paddle_dist():
            # 由于使用driver时模型一定会被初始化，因此在一开始程序一定会占用一部分显存来存放模型，然而这部分显存没有
            # 发挥任何作用。
            logger.warning(f"The program will use some extra space on {paddle.device.get_device()} to place your model since the model "
                            "has already been initialized.")

        self.output_from_new_proc = kwargs.get("output_from_new_proc", "only_error")
        assert isinstance(self.output_from_new_proc, str), "Parameter `output_from_new_proc` can only be `str` type."
        if self.output_from_new_proc not in {"all", "ignore", "only_error"}:
            os.makedirs(name=self.output_from_new_proc, exist_ok=True)
            self.output_from_new_proc = os.path.abspath(self.output_from_new_proc)

    def setup(self):
        """
        在主进程拉起其它子进程，将主进程作为rank 0
        """
        if self._has_setup:
            return
        self._has_setup = True
        # 如果用户需要使用多机模式，那么一定进入到这里；
        if self.is_pull_by_paddle_run:

            if self.outside_fleet:
                # 已经初始化了多机环境
                self.set_from_fleet_environment()
            else:
                # 用户没有初始化多机环境
                # TODO 绕一下
                # dist.get_world_size() 只能在初始化之后进行调用；
                self.world_size = int(os.environ.get("PADDLE_TRAINERS_NUM"))
                self.global_rank = int(os.environ.get("PADDLE_TRAINER_ID"))
                reset_seed()
                logger.info(f"\nworld size, global rank: {self.world_size}, {self.global_rank}\n")
                if not parallel_helper._is_parallel_ctx_initialized():
                    fleet.init(self.role_maker, self.is_collective, self.strategy)

                os.environ["fastnlp_paddle_launch_not_fleet"] = "yes"

        else:
            # 在用户只使用了一个分布式 trainer 的情况下
            # 此时 parallel_helper._is_parallel_ctx_initialized() 一定为 False
            # parallel_device 是 list，
            if not parallel_helper._is_parallel_ctx_initialized():
                # 没有初始化分布式环境，且是主进程
                self.init_fleet_and_set()
            # 用户在这个 trainer 前面又初始化了一个 trainer，并且使用的是 PaddleFleetDriver；
            else:
                # 已经设置过一次，保证参数必须是一样的
                pre_gpus = os.environ[FASTNLP_DISTRIBUTED_CHECK]
                pre_gpus = [int (x) for x in pre_gpus.split(",")]
                if sorted(pre_gpus) != sorted(self.parallel_device):
                    raise RuntimeError("Notice you are using `PaddleFleetDriver` after one instantiated `PaddleFleetDriver`, it is not"
                                    "allowed that your second `PaddleFleetDriver` has a new setting of parameters `parallel_device`.")
                self.world_size = dist.get_world_size()
                self.global_rank = dist.get_rank()

        if not self.outside_fleet:
            # self.model.to(self.model_device)
            self.configure_fleet()

        self.barrier()

        # 初始化 self._pids，从而使得每一个进程都能接受到 rank0 的 send 操作；
        # TODO 不用.to会怎么样？
        self._pids = []
        dist.all_gather(self._pids, paddle.to_tensor(os.getpid(), dtype="int32"))
        # TODO LOCAL_WORLD_SIZE
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE")) if "LOCAL_WORLD_SIZE" in os.environ else None
        if local_world_size is None:
            local_world_size = paddle.to_tensor(self.local_rank, dtype="int32")
            dist.all_reduce(local_world_size, op=dist.ReduceOp.MAX)
            local_world_size = local_world_size.item() + 1

        node_rank = self.global_rank // local_world_size
        self._pids = self._pids[node_rank*local_world_size: (node_rank+1)*local_world_size]
        self._pids = self.tensor_to_numeric(self._pids)

    def init_fleet_and_set(self):
        """
        使用 FleetLauncher 拉起子进程
        """
        if self.local_rank == 0:
            # 是 rank0 的话，则拉起其它子进程
            print("in launcher")
            launcher = FleetLauncher(self.parallel_device, self.output_from_new_proc)
            launcher.launch()
        # 设置参数和初始化分布式环境
        fleet.init(self.role_maker, self.is_collective, self.strategy)
        self.global_rank = int(os.getenv("PADDLE_TRAINER_ID"))
        self.world_size = int(os.getenv("PADDLE_TRAINERS_NUM"))

        # 正常情况下不会Assert出问题，但还是保险一下
        assert self.global_rank is not None
        assert self.world_size is not None
        assert self.world_size == len(self.parallel_device)

    def set_from_fleet_environment(self):
        """
        当用户使用了 `python -m paddle.distributed.launch xxx.py` 启动时，我们需要
        根据 paddle 设置的环境变量来获得各种属性
        """
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()

    def barrier(self):
        dist.barrier()

    def configure_fleet(self):
        if not self._configured and not isinstance(self.model, DataParallel):
            self.model = DataParallel(
                _FleetWrappingModel(self.model),
                **self._fleet_kwargs
            )

            self._train_step = partial(self.model, **{_MODE_PARAMETER: ForwardState.TRAIN})
            self._validate_step = partial(self.model, **{_MODE_PARAMETER: ForwardState.VALIDATE})
            self._test_step = partial(self.model, **{_MODE_PARAMETER: ForwardState.TEST})

        self._configured = True

    @property
    def world_size(self) -> int:
        return self._world_size

    @world_size.setter
    def world_size(self, size: int) -> None:
        self._world_size = size

    @property
    def global_rank(self) -> int:
        return self._global_rank

    @global_rank.setter
    def global_rank(self, rank: int) -> None:
        self._global_rank = rank

    @property
    def local_rank(self) -> int:
        return int(os.getenv("PADDLE_RANK_IN_NODE", "0"))

    @property
    def model_device(self):
        return self._model_device

    @property
    def data_device(self):
        if self.outside_fleet:
            return self._data_device
        return self.model_device

    def train_step(self, batch):
        return self._train_step(batch)

    def validate_step(self, batch):
        return self._validate_step(batch)

    def test_step(self, batch):
        return self._test_step(batch)

    def set_dist_repro_dataloader(self, dataloader, dist: Optional[Union[str, ReproducibleSampler, RandomBatchSampler]],
                                  reproducible: bool = False, sampler_or_batch_sampler=None):
        # 暂时不支持iterableDataset
        assert dataloader.dataset_kind != _DatasetKind.ITER, \
                    "FastNLP does not support `IteratorDataset` now."
        # 如果 dist 为 ReproducibleBatchSampler, ReproducibleSampler 说明是在断点重训时 driver.load 函数调用；
        # 注意这里不需要调用 dist_sampler.set_distributed；因为如果用户使用的是 TorchDDPDriver，那么其在 Trainer 初始化的时候就已经调用了该函数；
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
                raise RuntimeError("It is not allowed to use checkpoint retraining when you initialize ddp out of our "
                                   "control.")
            else:
                if isinstance(dist, ReproducibleBatchSampler):
                    dist = re_instantiate_sampler(dist)
                    return replace_batch_sampler(dataloader, dist)
                if isinstance(dist, ReproducibleSampler):
                    dist = re_instantiate_sampler(dist)
                    return replace_sampler(dataloader, dist)
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
            if isinstance(args.sampler, ReproducibleSampler):
                sampler = conversion_between_reproducible_and_unrepeated_sampler(args.sampler)
            elif not isinstance(args.sampler, UnrepeatedSampler):
                sampler = UnrepeatedSequentialSampler(
                    dataset=args.dataset
                )
            else:
                sampler = re_instantiate_sampler(args.sampler)
            sampler.set_distributed(
                num_replicas=self.world_size,
                rank=self.global_rank
            )
            return replace_sampler(dataloader, sampler)
        else:
            raise ValueError("Parameter `dist_sampler` can only be one of three values: ('dist', 'unrepeatdist', None).")

    def backward(self, loss):
        self.grad_scaler.scale(loss).backward()

    def step(self):
        for optimizer in self.optimizers:
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()

    def is_global_zero(self):
        return self.global_rank == 0

    def get_no_sync_context(self):
        return self.model.no_sync

    def unwrap_model(self):
        _layers = self.model._layers
        if isinstance(_layers, _FleetWrappingModel):
            return _layers.model
        else:
            return _layers

    def get_local_rank(self) ->int:
        return self.local_rank

    def is_distributed(self):
        return True

    def move_data_to_device(self, batch: 'paddle.Tensor'):
        device = self.data_device
        # 因为设置了CUDA_VISIBLE_DEVICES，可能会引起错误
        device = get_device_from_visible(device)
        return paddle_move_data_to_device(batch, device)

    @staticmethod
    def _check_optimizer_legality(optimizers):
        """
        paddle存在设置分布式optimizers的函数，返回值为fleet.meta_optimizers.HybridParallelOptimizer
        重写是为了防止单卡下也传入了分布式的优化器
        """
        DistribuedOptimizer = fleet.meta_optimizers.HybridParallelOptimizer
        for each_optimizer in optimizers:
            if not isinstance(each_optimizer, (Optimizer, DistribuedOptimizer)):
                raise ValueError(f"Each optimizer of parameter `optimizers` should be 'paddle.optimizer.Optimizer' type, "
                                f"not {type(each_optimizer)}.")
