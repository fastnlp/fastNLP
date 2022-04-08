import os
import sys
import __main__
import socket
import numpy as np
from time import sleep
from typing import List, Optional, Union, Dict
from functools import partial

# todo 这个等大家的 __all__ 都弄完后改为 from fastNLP.env import；
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

__all__ = [
    'TorchDDPDriver'
]

from .torch_driver import TorchDriver
from fastNLP.core.drivers.torch_driver.utils import (
    _DDPWrappingModel,
    ForwardState,
    _MODE_PARAMETER,
    reset_seed,
    replace_sampler
)
from fastNLP.core.drivers.utils import distributed_open_proc
from fastNLP.core.utils import auto_param_call, check_user_specific_params
from fastNLP.core.samplers import ReproducibleIterator, RandomSampler, UnrepeatedDistributedSampler
from fastNLP.envs import FASTNLP_DISTRIBUTED_CHECK, FASTNLP_GLOBAL_RANK, FASTNLP_GLOBAL_SEED
from fastNLP.core.log import logger
from fastNLP.core.drivers.torch_driver.dist_utils import fastnlp_torch_all_gather, fastnlp_torch_broadcast_object
from fastNLP.core.samplers import re_instantiate_sampler


class TorchDDPDriver(TorchDriver):
    def __init__(
            self,
            model,
            parallel_device: Optional[Union[List["torch.device"], "torch.device"]],
            is_pull_by_torch_run: bool = False,
            fp16: bool = False,
            **kwargs
    ):
        """
        DDP 目前考虑支持的三种启动方式：
        1. 用户自己不进行 ddp 的任何操作，直接使用我们的 Trainer，并且只运行一个 main 脚本，这时是由我们自己使用 open_subprocesses 拉起
         多个进程，然后 TorchDDPDriver 自己 init_process_group；
        2. 其它情况同 1，但是用户自己使用 python -m torch.distributed.launch 拉起；
        3. 用户自己在外面初始化 DDP，并且通过 python -m torch.distributed.launch 拉起；

        注意多机的启动强制要求用户在每一台机器上使用 python -m torch.distributed.launch 启动；

        如果用户自己在外面初始化了 ddp，那么
            parallel_device 为 None；
            data_device 为 表示单卡的一个参数；
            dist.is_initialized 为 true；

        """
        super(TorchDDPDriver, self).__init__(model, fp16=fp16, **kwargs)

        if isinstance(model, torch.nn.DataParallel):
            raise ValueError(f"Parameter `model` can not be `DataParallel` in `TorchDDPDriver`, it should be "
                             f"`torch.nn.Module` or `torch.nn.parallel.DistributedDataParallel` type.")

        # 如果用户自己在外面初始化 DDP，那么其一定是通过 python -m torch.distributed.launch 拉起的；
        self.is_pull_by_torch_run = is_pull_by_torch_run
        self.parallel_device = parallel_device
        if not is_pull_by_torch_run and parallel_device is None:
            raise ValueError("Parameter `parallel_device` can not be None when using `TorchDDPDriver`. This error is caused "
                             "when your value of parameter `device` is `None` in your `Trainer` instance.")

        # 注意我们在 initialize_torch_driver 中的逻辑就是如果是 is_pull_by_torch_run，那么我们就直接把 parallel_device 置为当前进程的gpu；
        if is_pull_by_torch_run:
            self.model_device = parallel_device
        else:
            # 我们的 model_device 一定是 torch.device，而不是一个 list；
            self.model_device = parallel_device[self.local_rank]

        # 如果用户自己在外面初始化了 DDP；
        self.outside_ddp = False
        if dist.is_initialized() and FASTNLP_DISTRIBUTED_CHECK not in os.environ and "fastnlp_special" not in os.environ:
            # 如果用户自己在外面初始化了 DDP，那么我们要求用户传入的模型一定是已经由 DistributedDataParallel 包裹后的模型；
            if not isinstance(model, DistributedDataParallel):
                raise RuntimeError(
                    "It is not allowed to input a normal model instead of `DistributedDataParallel` when"
                    "you initialize the ddp process out of our control.")

            self.outside_ddp = True
            # 用户只有将模型上传到对应机器上后才能用 DistributedDataParallel 包裹，因此如果用户在外面初始化了 DDP，那么在 TorchDDPDriver 中
            #  我们就直接将 model_device 置为 None；
            self.model_device = None

            def _running_fn_(batch, step_fn, signature_fn):
                if isinstance(batch, Dict):
                    return auto_param_call(step_fn, batch, signature_fn=signature_fn)
                else:
                    return self._validate_step(batch)

            model = model.module
            if hasattr(model, "train_step"):
                logger.warning(
                    "Notice your model is a `DistributedDataParallel` model. And your "
                    "model also implements the `train_step` method, which we can not call actually, we will"
                    " call `forward` function instead of `train_step` and you should note that.")
            self._train_step = partial(_running_fn_, step_fn=self.model, signature_fn=model.forward)
            # self._train_signature_fn = model.forward

            if hasattr(model, "validate_step"):
                logger.warning(
                    "Notice your model is a `DistributedDataParallel` model. And your "
                    "model also implements the `validate_step` method, which we can not call actually, "
                    "we will call `forward` function instead of `validate_step` and you should note that.")
            self._validate_step = partial(_running_fn_, step_fn=self.model, signature_fn=model.forward)
            # self._validate_signature_fn = model.forward

            if hasattr(model, "test_step"):
                logger.warning(
                    "Notice your model is a `DistributedDataParallel` model. And your "
                    "model also implements the `test_step` method, which we can not call actually, we will"
                    " call `forward` function instead of `test_step` and you should note that.")
            self._test_step = partial(_running_fn_, step_fn=self.model, signature_fn=model.forward)
            # self._test_signature_fn = model.forward

        # 当用户自己在外面初始化 DDP 时我们会将 model_device 置为 None，这是用户可以通过 `data_device` 将对应的数据移到指定的机器上;
        self._data_device = kwargs.get("data_device", None)
        # if self.outside_ddp and self._data_device is None:
        #     raise RuntimeError("When you initialize your ddp out of our control, the parameter "
        #                        "`data_device` can not be None.")
        if isinstance(self._data_device, int):
            if self._data_device < 0:
                raise ValueError("Parameter `data_device` can not be smaller than 0.")
            _could_use_device_num = torch.cuda.device_count()
            if self._data_device >= _could_use_device_num:
                raise ValueError("The gpu device that parameter `device` specifies is not existed.")
            self._data_device = torch.device(f"cuda:{self._data_device}")
        elif isinstance(self._data_device, str):
            self._data_device = torch.device(self._data_device)
        elif self._data_device is not None and not isinstance(self._data_device, torch.device):
            raise ValueError("Parameter `device` is wrong type, please check our documentation for the right use.")

        self._master_port = None
        # world_size 表示的就是全局的显卡的数量；
        self.world_size = None  # int(os.environ.get("WORLD_SIZE"))  len(self.parallel_device)
        self.global_rank = 0
        self._configured = False  # 防止重复调用 configure_ddp() 函数使用的

        self._ddp_kwargs = kwargs.get("torch_ddp_kwargs", {})
        check_user_specific_params(self._ddp_kwargs, DistributedDataParallel.__init__)
        if len(self.model._buffers) != 0 and self._ddp_kwargs.get("broadcast_buffers", None) is None:
            logger.info("Notice your model has buffers and you are using `TorchDDPDriver`, but you do not set "
                        "'broadcast_buffers' in your trainer. Cause in most situations, this parameter can be set"
                        " to 'False' to avoid redundant data translation between different processes.")

        self.output_from_new_proc = kwargs.get("output_from_new_proc", "only_error")
        assert isinstance(self.output_from_new_proc, str), "Parameter `output_from_new_proc` can only be `str` type."
        if self.output_from_new_proc not in {"all", "ignore", "only_error"}:
            os.makedirs(name=self.output_from_new_proc, exist_ok=True)
            self.output_from_new_proc = os.path.abspath(self.output_from_new_proc)

        # 设置这一参数是因为 evaluator 中也会进行 setup 操作，但是显然是不需要的也不应该的；
        self._has_setup = False

    def setup(self):
        if self._has_setup:
            return
        self._has_setup = True
        # 如果用户需要使用多机模式，那么一定进入到这里；
        if self.is_pull_by_torch_run:

            if self.outside_ddp:
                self.world_size = dist.get_world_size()
                self.global_rank = dist.get_rank()
            else:
                # dist.get_world_size() 只能在 dist.init_process_group 初始化之后进行调用；
                self.world_size = int(os.environ.get("WORLD_SIZE"))
                self.global_rank = int(os.environ.get("RANK"))
                reset_seed()
                logger.info(f"World size:{self.world_size}, Global rank:{self.global_rank}")

                if not dist.is_initialized():
                    dist.init_process_group(
                        backend="nccl", rank=self.global_rank, world_size=self.world_size
                    )

                os.environ["fastnlp_special"] = "yes"

        # 进入到这里的情况时：
        # dist.is_initialized 一定为 False；
        # 一定是单机；
        # self.parallel_device 一定是 List[torch.device]；
        else:
            if not dist.is_initialized():
                # 这里主要的问题在于要区分 rank0 和其它 rank 的情况；
                self.world_size = len(self.parallel_device)
                self.open_subprocess()
                self.global_rank = self.local_rank  # rank 一定是通过环境变量去获取的；
                reset_seed()
                dist.init_process_group(
                    backend="nccl", rank=self.global_rank, world_size=self.world_size
                )
            # 用户在这个 trainer 前面又初始化了一个 trainer，并且使用的是 TorchDDPDriver；
            else:
                # 如果 `dist.is_initialized() == True`，那么说明 TorchDDPDriver 在之前已经初始化并且已经 setup 过一次，那么我们需要保证现在
                #  使用的（即之后的）TorchDDPDriver 的设置和第一个 TorchDDPDriver 是完全一样的；
                pre_num_processes = int(os.environ[FASTNLP_DISTRIBUTED_CHECK])
                if pre_num_processes != len(self.parallel_device):
                    raise RuntimeError("Notice you are using `TorchDDPDriver` after one instantiated `TorchDDPDriver`, it is not"
                                       "allowed that your second `TorchDDPDriver` has a new setting of parameters "
                                       "`num_nodes` and `num_processes`.")
                self.world_size = dist.get_world_size()
                self.global_rank = dist.get_rank()

        if not self.outside_ddp:
            torch.cuda.set_device(self.model_device)
            self.model.to(self.model_device)
            self.configure_ddp()

        self.barrier()
        # 初始化 self._pids，从而使得每一个进程都能接受到 rank0 的 send 操作；
        self._pids = [torch.tensor(0, dtype=torch.int).to(self.data_device) for _ in range(dist.get_world_size())]
        dist.all_gather(self._pids, torch.tensor(os.getpid(), dtype=torch.int).to(self.data_device))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE")) if "LOCAL_WORLD_SIZE" in os.environ else None
        if local_world_size is None:
            local_world_size = torch.tensor(int(os.environ.get("LOCAL_RANK")), dtype=torch.int).to(self.data_device)
            dist.all_reduce(local_world_size, op=dist.ReduceOp.MAX)
            local_world_size = local_world_size.tolist() + 1

        node_rank = self.global_rank // local_world_size
        self._pids = self._pids[node_rank*local_world_size: (node_rank+1)*local_world_size]
        self._pids = self.tensor_to_numeric(self._pids)

    def configure_ddp(self):
        if not self._configured and not isinstance(self.model, DistributedDataParallel):
            self.model = DistributedDataParallel(
                # 注意这里的 self.model_device 是 `torch.device` type，因此 self.model_device.index；
                _DDPWrappingModel(self.model), device_ids=[self.model_device.index],
                **self._ddp_kwargs
            )

            self._train_step = partial(self.model, **{_MODE_PARAMETER: ForwardState.TRAIN})
            self._validate_step = partial(self.model, **{_MODE_PARAMETER: ForwardState.VALIDATE})
            self._test_step = partial(self.model, **{_MODE_PARAMETER: ForwardState.TEST})

        self._configured = True

    def open_subprocess(self):
        if self.local_rank == 0:
            # self._consensus_file = Path(tempfile.mkstemp()[1])
            # self._consensus_file.unlink()

            # Script called as `python a/b/c.py`
            if __main__.__spec__ is None:  # pragma: no-cover
                # pull out the commands used to run the script and resolve the abs file path
                command = sys.argv
                command[0] = os.path.abspath(command[0])
                # use the same python interpreter and actually running
                command = [sys.executable] + command
            # Script called as `python -m a.b.c`
            else:
                command = [sys.executable, "-m", __main__.__spec__._name] + sys.argv[1:]

            os.environ['MASTER_ADDR'] = self.master_address
            os.environ['MASTER_PORT'] = self.master_port

            os.environ["LOCAL_RANK"] = str(self.local_rank)
            os.environ["WORLD_SIZE"] = f"{self.world_size}"

            os.environ[FASTNLP_DISTRIBUTED_CHECK] = f"{len(self.parallel_device)}"
            os.environ[FASTNLP_GLOBAL_RANK] = "0"

            interactive_ddp_procs = []

            for rank in range(1, len(self.parallel_device)):
                env_copy = os.environ.copy()
                env_copy["LOCAL_RANK"] = f"{rank}"

                # 如果是多机，一定需要用户自己拉起，因此我们自己使用 open_subprocesses 开启的进程的 FASTNLP_GLOBAL_RANK 一定是 LOCAL_RANK；
                env_copy[FASTNLP_GLOBAL_RANK] = str(rank)

                proc = distributed_open_proc(self.output_from_new_proc, command, env_copy, self.global_rank)

                interactive_ddp_procs.append(proc)
                delay = np.random.uniform(1, 5, 1)[0]
                sleep(delay)

    @property
    def master_address(self) -> str:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    @property
    def master_port(self) -> str:
        if self.outside_ddp:
            return os.environ.get("MASTER_PORT")
        if self._master_port is None:
            self._master_port = os.environ.get("MASTER_PORT", find_free_network_port())
        return self._master_port

    @property
    def world_size(self) -> int:
        return self._world_size

    @world_size.setter
    def world_size(self, size: int):
        self._world_size = size

    @property
    def global_rank(self) -> int:
        return self._global_rank

    @global_rank.setter
    def global_rank(self, rank: int) -> None:
        self._global_rank = rank

    @property
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def data_device(self):
        if self.outside_ddp:
            return self._data_device
        return self.model_device

    def train_step(self, batch):
        # 注意这里的 self.model 已经是 'fastNLP.drivers.utils._DDPWrappingModel'；
        # return self.model(batch, **{_MODE_PARAMETER: ForwardState.TRAIN})
        return self._train_step(batch)

    def validate_step(self, batch):
        # return self.model(batch, **{_MODE_PARAMETER: ForwardState.VALIDATE})
        return self._validate_step(batch)

    def test_step(self, batch):
        # return self.model(batch, **{_MODE_PARAMETER: ForwardState.TEST})
        return self._test_step(batch)

    def replace_sampler(self, dataloader, dist_sampler: Optional[Union[str, ReproducibleIterator]] = "dist", reproducible: bool = False):
        if isinstance(dist_sampler, ReproducibleIterator):
            # 注意这里不需要调用 dist_sampler.set_distributed；因为如果用户使用的是 TorchDDPDriver，那么其在 Trainer 初始化的时候就已经调用了该函数；
            dist_sampler = re_instantiate_sampler(dist_sampler)
            return replace_sampler(dataloader, dist_sampler)

        # trainer, evaluator
        if dist_sampler is None:
            if reproducible:
                raise RuntimeError("It is not allowed to use checkpoint retraining when you initialize ddp out of our "
                                   "control.")
            else:
                return dataloader
        # trainer
        elif dist_sampler == "dist":
            args = self.get_dataloader_args(dataloader)
            # 如果用户的 trainer.use_dist_sampler 为 True，那么此时其是否进行断点重训，不影响这里的行为；
            if isinstance(args.sampler, ReproducibleIterator):
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
                # todo 这个你写个todo吧，有两个角度；第一个是dataloader即使检测到sampler是我们reproducible，也不能直接set_distributeds; 第二个如果是单卡的，也需要替换sampler乃至切换sampler的状态，方式之前多卡，现在切换成单卡运行
                sampler.set_distributed(
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                    pad=True
                )
                return replace_sampler(dataloader, sampler)

        # evaluator
        elif dist_sampler == "unrepeatdist":
            args = self.get_dataloader_args(dataloader)
            sampler = UnrepeatedDistributedSampler(
                dataset=args.dataset,
                shuffle=args.shuffle,
            )
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
        # 注意此时的 model 是 "DistributedDataParallel" 对象；
        return self.model.no_sync

    def unwrap_model(self):
        _module = self.model.module
        if isinstance(_module, _DDPWrappingModel):
            return _module.model
        else:
            return _module

    def get_local_rank(self) -> int:
        return self.local_rank

    def barrier(self):
        torch.distributed.barrier(async_op=True)

    def is_distributed(self):
        return True

    def broadcast_object(self, obj, src:int=0, group=None, **kwargs):
        """
        从 src 端将 obj 对象（可能是 tensor ，可能是 object ）发送到 dst 处。如果是非 tensor 的对象会尝试使用 pickle 进行打包进行
            传输，然后再 dst 处再加载回来。仅在分布式的 driver 中有实际意义。

        :param obj: obj，可能是 Tensor 或 嵌套类型的数据
        :param int src: source 的 global rank 。
        :param int dst: target 的 global rank，可以是多个目标 rank
        :param group: 所属的 group
        :param kwargs:
        :return: 如果当前不是分布式 driver 直接返回输入的 obj 。如果当前 rank 是接收端（其 global rank 包含在了 dst 中），则返回
            接收到的参数；如果是 source 端则返回发射的内容；既不是发送端、又不是接收端，则返回 None 。
        """
        return fastnlp_torch_broadcast_object(obj, src, device=self.data_device, group=group)

    def all_gather(self, obj, group) -> List:
        """
        将 obj 互相传送到其它所有的 rank 上，其中 obj 可能是 Tensor，也可能是嵌套结构的 object 。如果不是基础类型的数据，尝试通过
            pickle 进行序列化，接收到之后再反序列化。

        example:
            obj = {
                'a': [1, 1],
                'b': [[1, 2], [1, 2]],
                'c': {
                    'd': [1, 2]
                }
            }
            ->
            [
                {'a': 1, 'b':[1, 2], 'c':{'d': 1}},
                {'a': 1, 'b':[1, 2], 'c':{'d': 2}}
            ]

        :param obj: 需要传输的对象，在每个rank上都应该保持相同的结构。
        :param group:
        :return:
        """
        return fastnlp_torch_all_gather(obj, device=self.data_device, group=group)


def find_free_network_port() -> str:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real master node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return str(port)
