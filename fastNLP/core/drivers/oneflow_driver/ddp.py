import os
from typing import List, Optional, Union, Dict

from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW

if _NEED_IMPORT_ONEFLOW:
    import oneflow
    import oneflow.comm as comm
    import oneflow.env as dist_env
    from oneflow.nn.parallel import DistributedDataParallel
    from oneflow.utils.data import BatchSampler

__all__ = [
    "OneflowDDPDriver"
]

from .oneflow_driver import OneflowDriver
from fastNLP.core.drivers.oneflow_driver.utils import (
    replace_sampler,
    replace_batch_sampler
)
from fastNLP.core.utils import check_user_specific_params
from fastNLP.core.samplers import ReproducibleSampler, RandomSampler, UnrepeatedSequentialSampler, \
    ReproducibleBatchSampler, \
    re_instantiate_sampler, UnrepeatedSampler, conversion_between_reproducible_and_unrepeated_sampler
from fastNLP.envs import FASTNLP_GLOBAL_SEED, FASTNLP_NO_SYNC
from fastNLP.core.log import logger
from fastNLP.core.drivers.oneflow_driver.dist_utils import fastnlp_oneflow_all_gather, fastnlp_oneflow_broadcast_object
from .utils import _check_dataloader_args_for_distributed


class OneflowDDPDriver(OneflowDriver):
    r"""
    ``OneflowDDPDriver`` 实现了动态图下使用 ``DistributedDataParallel`` 进行的数据并行分布式训练。

    .. note::

        您在绝大多数情况下不需要自己使用到该类，通过向 ``Trainer`` 传入正确的参数，您可以方便快速地部署您的分布式训练。

        ``OneflowDDPDriver`` 目前支持两种启动方式：
        
            1. 用户不做任何处理，通过运行 ``python -m oneflow.distributed.launch --nproc_per_node 2 train.py`` 启动；
            2. 用户将模型通过 ``DistributedDataParallel`` 处理后，通过运行 ``python -m oneflow.distributed.launch --nproc_per_node 2 train.py`` 启动；

        注意多机的启动强制要求用户在每一台机器上使用 ``python -m oneflow.distributed.launch`` 启动；因此我们不会在 ``OneflowDDPDriver`` 中保存
        任何当前有多少台机器的信息。

    :param model: 传入给 ``Trainer`` 的 ``model`` 参数
    :param parallel_device: 该参数无效，**fastNLP** 会自动获取当前进程的设备
    :param fp16: 是否开启 fp16 训练；目前该参数无效
    :param oneflow_kwargs: 
        * *ddp_kwargs* -- 用于 ``DistributedDataParallel`` 的其它参数，详情可查阅 **oneflow** 的官方文档
    :kwargs:
        * *model_wo_auto_param_call* (``bool``) -- 是否关闭在训练时调用我们的 ``auto_param_call`` 函数来自动匹配 batch 和前向函数的参数的行为

        .. note::

            关于该参数的详细说明，请参见 :class:`~fastNLP.core.controllers.Trainer` 中的描述；函数 ``auto_param_call`` 详见 :func:`fastNLP.core.utils.auto_param_call`。

    """

    def __init__(
            self,
            model,
            parallel_device: Optional["oneflow.device"],
            fp16: bool = False,
            oneflow_kwargs: Dict = None,
            **kwargs
    ):

        super(OneflowDDPDriver, self).__init__(model, fp16=fp16, oneflow_kwargs=oneflow_kwargs, **kwargs)

        # oneflow 会自己初始化通信组，因此 parallel_device 实际上不起作用，可以通过 current_device 获取设备
        self.model_device = oneflow.device("cuda", oneflow.cuda.current_device())
        self._data_device = self.model_device

        self.global_rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        self._ddp_kwargs = self._oneflow_kwargs.get("ddp_kwargs", {})
        check_user_specific_params(self._ddp_kwargs, DistributedDataParallel.__init__, DistributedDataParallel.__name__)
        if len(self.model._buffers) != 0 and self._ddp_kwargs.get("broadcast_buffers", None) is None:
            logger.info("Notice your model has buffers and you are using `OneflowDDPDriver`, but you do not set "
                        "'broadcast_buffers' in your trainer. Cause in most situations, this parameter can be set"
                        " to 'False' to avoid redundant data communication between different processes.")

        self.output_from_new_proc = kwargs.get("output_from_new_proc", "only_error")
        assert isinstance(self.output_from_new_proc, str), "Parameter `output_from_new_proc` can only be `str` type."
        if self.output_from_new_proc not in {"all", "ignore", "only_error"}:
            os.makedirs(name=self.output_from_new_proc, exist_ok=True)
            self.output_from_new_proc = os.path.abspath(self.output_from_new_proc)

        self._has_setup = False  # 设置这一参数是因为 evaluator 中也会进行 setup 操作，但是显然是不需要的也不应该的；
        self._has_ddpwrapped = False# hasattr(model, )

    def setup(self):
        r"""
        将模型用 ``DistributedDataParallel`` 进行处理。
        """
        if self._has_setup:
            return
        self._has_setup = True

        self.configure_ddp()
        self.barrier()
        # 初始化 self._pids，从而使得每一个进程都能接受到 rank0 的 send 操作；
        # self._pids = [oneflow.tensor(0, dtype=oneflow.int).to(self.data_device) for _ in range(dist_env.get_world_size())]
        # comm.all_gather(self._pids, oneflow.tensor(os.getpid(), dtype=oneflow.int).to(self.data_device))
        # local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE")) if "LOCAL_WORLD_SIZE" in os.environ else None
        # if local_world_size is None:
        #     local_world_size = oneflow.tensor(int(os.environ.get("LOCAL_RANK")), dtype=oneflow.int).to(self.data_device)
        #     comm.all_reduce(local_world_size, op=dist_env.ReduceOp.MAX)
        #     local_world_size = local_world_size.tolist() + 1

        # node_rank = self.global_rank // local_world_size
        # self._pids = self._pids[node_rank * local_world_size: (node_rank + 1) * local_world_size]
        # self._pids = self.tensor_to_numeric(self._pids)

    def configure_ddp(self):
        if not hasattr(self.model, "_ddp_state_for_reversed_params"):
            self.model.to(self.model_device)
            self.model = DistributedDataParallel(
                # 注意这里的 self.model_device 是 `oneflow.device` type，因此 self.model_device.index；
                self.model,
                **self._ddp_kwargs
            )
            self._has_ddpwrapped = True

    @property
    def master_address(self) -> str:
        """
        分布式训练中的地址 ``MASTER_ADDR``
        """
        return os.environ.get("MASTER_ADDR")

    @property
    def master_port(self) -> str:
        """
        分布式训练使用的端口 ``MASTER_PORT``
        """
        return os.environ.get("MASTER_PORT")

    @property
    def world_size(self) -> int:
        """
        分布式训练的进程总数 ``WORLD_SIZE``
        """
        return self._world_size

    @world_size.setter
    def world_size(self, size: int):
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
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def data_device(self):
        """
        数据所在的设备。由于 **oneflow** 可以通过 :func:`oneflow.cuda.current_device` 获取当前进程的设备，因此
        该属性和 ``model_device`` 表现相同。
        """
        return self._data_device

    def set_dist_repro_dataloader(self, dataloader,
                                  dist: Optional[Union[str, ReproducibleSampler, ReproducibleBatchSampler]] = None,
                                  reproducible: bool = False):
        # 如果 dist 为 ReproducibleBatchSampler, ReproducibleSampler 说明是在断点重训时 driver.load_checkpoint 函数调用；
        # 注意这里不需要调用 dist_sampler.set_distributed；因为如果用户使用的是 OneflowDDPDriver，那么其在 Trainer 初始化的时候就已经调用了该函数；
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
                    return replace_batch_sampler(dataloader, re_instantiate_sampler(args.batch_sampler))
                if isinstance(args.sampler, ReproducibleSampler):
                    return replace_sampler(dataloader, re_instantiate_sampler(args.sampler))
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
                _check_dataloader_args_for_distributed(args, controller="Trainer")
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
            raise ValueError(
                "Parameter `dist_sampler` can only be one of three values: ('dist', 'unrepeatdist', None).")

    def is_global_zero(self):
        r"""
        :return: 当前的进程是否在全局上是进程 0 
        """
        return self.global_rank == 0

    def get_model_no_sync_context(self):
        r"""
        :return: 一个 ``context`` 上下文环境，用于关闭各个进程之间的同步；该功能暂时无效，返回一个空的上下文环境
        """
        # TODO 暂时没有在 oneflow 中找到类似的功能；
        from fastNLP.core.utils import nullcontext
        return nullcontext
        return self.model.no_sync

    def unwrap_model(self):
        r"""
        :return: 使用的原始模型
        """
        return self.model

    def get_local_rank(self) -> int:
        r"""
        :return: 当前进程局部的进程编号
        """
        return self.local_rank

    def barrier(self):
        r"""
        同步各个进程之间的操作
        """
        if int(os.environ.get(FASTNLP_NO_SYNC, 0)) < 1:  # 当 FASTNLP_NO_SYNC 小于 1 时实际执行
            comm.barrier()

    def is_distributed(self):
        r"""
        :return: 当前使用的 driver 是否是分布式的 driver，对于 ``OneflowDDPDriver`` 来说，该函数一定返回 ``True``
        """
        return True

    def broadcast_object(self, obj, src: int = 0, group=None, **kwargs):
        r"""
        从 ``src`` 端将 ``obj`` 对象（可能是 tensor ，可能是 object ）广播到其它进程。如果是非 tensor 的对象会尝试使用 pickle 进行打包进行
        传输，然后在接收处处再加载回来。仅在分布式的 driver 中有实际意义。

        :param obj: obj，可能是 Tensor 或 嵌套类型的数据
        :param src: 发送方的 ``global_rank``
        :param group: 该参数无效
        :return: 如果当前 rank 是接收端，则返回接收到的参数；如果是 source 端则返回发送的内容。如果环境变量 ``FASTNLP_NO_SYNC`` 为 **2** 则
            返回 ``None``
        """
        if int(os.environ.get(FASTNLP_NO_SYNC, 0)) == 2:  # 如果 FASTNLP_NO_SYNC == 2 直接返回。
            return
        return fastnlp_oneflow_broadcast_object(obj, src, device=self.data_device)

    def all_gather(self, obj) -> List:
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
        :param group: 该参数无效。
        :return: 所有 rank 发送的 ``obj`` 聚合在一起的内容；如果环境变量 ``FASTNLP_NO_SYNC`` 为 **2** 则不会执行，直接返回 ``[obj]`` 。
        """
        if int(os.environ.get(FASTNLP_NO_SYNC, 0)) == 2:  # 如果 FASTNLP_NO_SYNC 表示不执行
            return [obj]
        return fastnlp_oneflow_all_gather(obj)
