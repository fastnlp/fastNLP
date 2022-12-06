r"""
"""

r"""
`TorchDDPDriver` 目前支持的三种启动方式：
1. 用户自己不进行 ddp 的任何操作，直接使用我们的 Trainer，这时是由我们自己使用 `open_subprocesses` 拉起多个进程，
 然后 `TorchDDPDriver` 自己通过调用 `dist.init_process_group` 来初始化 ddp 的通信组；（情况 A）
2. 用户同样不在 Trainer 之外初始化 ddp，但是用户自己使用 python -m torch.distributed.launch 拉起来创建多个进程，这时我们仍旧
 会通过调用 `dist.init_process_group` 来初始化 ddp 的通信组；（情况 B）
3. 用户自己在外面初始化 DDP，并且通过 python -m torch.distributed.launch 拉起，这时无论是多个进程的拉起和 ddp 的通信组的建立
 都由用户自己操作，我们只会在 driver.setup 的时候对 `TorchDDPDriver` 设置一些必要的属性值；（情况 C）

注意多机的启动强制要求用户在每一台机器上使用 python -m torch.distributed.launch 启动；因此我们不会在 `TorchDDPDriver` 中保存
 任何当前有多少台机器的信息（num_nodes，不是 gpu 的数量）；

Part 1：三种启动方式的具体分析：
（1）对于用户运行的脚本中，如果 `driver.setup` 只会被调用一次（意味着用户的启动脚本中只初始化了一个 trainer/evaluator）时，
 `TorchDDPDriver` 在初始化以及 `setup` 函数中会做的事情分别如下所示：
    -> 情况 A：这种情况下用户传入的 model 在一定是普通的 model（没有经 `DistributedDataParallel` 包裹的model），
     因为 `DistributedDataParallel` 的使用一定要求 init_process_group 已经被调用用来建立当前的 ddp 通信组；但是这意味着如果
     用户需要使用 2 张以上的显卡，那么其必然需要使用 torch.distributed.launch 来启动，意味着就不是情况 A 了；
     这时我们首先会调用 `TorchDDPDriver.open_subprocess` 函数来拉起多个进程，其中进程的数量等于用户传入给 trainer 的使用的 gpu
     的数量（例如 `Trainer` 中的参数是 device=[0, 1, 6, 7]，那么我们就会使用第 0、1、6、7 张 gpu 来拉起 4 个进程）；
     接着我们会调用 `dist.init_process_group` 来初始化各个进程之间的通信组；
     这里需要注意拉起的新的进程会从前到后完整地运行一遍用户的启动脚本（例如 main.py），因此也都会运行这两个函数，但是需要注意只有进程 0
     才会去真正地运行 `TorchDDPDriver.open_subprocess`；进程 0 运行到 `dist.init_process_group`，pytorch 会阻塞进程 0 继续
     向前运行，直到其它进程也运行到这里；
     最后我们会设置这个进程对应的 device，然后将模型迁移到对应的机器上，再使用 `DistributedDataParallel` 将模型包裹；
     至此，ddp 的环境配置过程全部完成；

    -> 情况 B：注意这种情况我们直接限定了用户是通过 torch.distributed.launch 拉起，并且没有自己建立 ddp 的通信组。这时在
     `TorchDDPDriver` 的初始化和 setup 函数的调用过程中，与情况 A 首要的不同就在于用户在 trainer 中输入的参数 device 不再有效，
     这时每个进程所使用的 gpu 是我们直接通过 `torch.device("cuda:{local_rank}")` 来配置的；因此，如果用户想要实现使用特定 gpu
     设备的目的，可以通过自己设置环境变量实现（例如 os.environ["CUDA_VISIBLE_DEVICE"] 来实现）；剩下的操作和情况 A 类似；

    -> 情况 C：注意这种情况我们限定了用户是通过 torch.distributed.launch 拉起，并且 ddp 的通信组也是由自己建立。这时基本上所有的
     与操作相关的操作都应当由用户自己完成，包括迁移模型到对应 gpu 上以及将模型用 `DistributedDataParallel` 包裹等。
（2）如果 `driver.setup` 函数在脚本中会被调用两次及以上（意味着用户的启动脚本初始化了两个及以上的 trainer/evaluator）时：
注意这种情况下我们是会保证前后两个 trainer/evaluator 使用的 `TorchDDPDriver` 以及其初始化方式的一致性，换句话说，如果 trainer1
 检测到的启动方式是 '情况 A'，那么我们会保证 trainer2 检测到的启动方式同样是 '情况A'（即使这需要一些额外的处理）；因此这里我们主要讨论
 我们是通过怎样的操作来保证 trainer2/3/... 检测到的启动方式是和 trainer1 一致的；简单来说，我们是通过使用环境变量来标记每一种不同的
 启动方式来实现这一点的：
我们会使用 `FASTNLP_DISTRIBUTED_CHECK` 来标记 '情况 A'，使用 `fastnlp_torch_launch_not_ddp` 来标记 '情况 B'，意味着我们在
 使用 '情况 A' 来启动 `TorchDDPDriver` 时，我们会将 `FASTNLP_DISTRIBUTED_CHECK` 这一字符串注入到环境变量中，而 '情况 B' 时则
 会将 `fastnlp_torch_launch_not_ddp` 这一字符串注入到环境变量中。因此在 trainer2 的 `TorchDDPDriver` 的初始化和 setup 过程中，
 如果检测到这些特殊的环境变量，我们就会将启动方式变更为其对应的启动方式，即使其它的参数特征属于另外的启动方式。

Part 2：对应的代码细节：
    1. 如何判断当前的各进程之间的通信组已经被建立（ddp 已经被初始化）；
        dist.is_initialized()；
    2. 如何判断不同的进程是否是由 `python -m torch.distributed.launch` 拉起还是由我们的 `TorchDDPDriver.open_subprocess`
     函数拉起；
        我们会在用户脚本 `import fastNLP` 的时候检测当前的环境变量中是否有 'LOCAL_RANK'、'WORLD_SIZE' 以及没有 `FASTNLP_DISTRIBUTED_CHECK`，
        如果满足条件，则我们会向环境变量中注入特殊的值 'FASTNLP_BACKEND_LAUNCH' 来标记用户是否使用了 `python -m torch.distributed.launch`
        来拉起多个进程；
    3. 整体的处理判断流程：
         ___________________________________
        ｜进入 TorchDDPDriver 的 __init__ 函数｜
         ———————————————————————————————————
                         ↓
    ___________________________________________________
   ｜ 判断不同的进程是否是由 torch.distributed.launch 拉起 ｜
   ｜（或者我们自己的 open_subprocess 函数拉起）           ｜  -------------->
    ———————————————————————————————————————————————————                　｜
                         ↓ 是由 torch.distributed.launch 拉起            ｜ 我们自己的 open_subprocess 函数拉起多个进程
            　___________________________             　　　　　　　　　　　｜　
     ←←←←←  ｜ 检测用户是否自己初始化了 ddp ｜         　　　　　　　　　　　　　｜
     ↓       ———————————————————————————　　　　　　　　　　　　　　　　　　　 ↓
     ↓                   ↓ 是                                         ________
     ↓                  ______                                      ｜ 情况 A ｜
     ↓ 否               |情况 C|                                      —————————
     ↓                 ———————
     ↓
     ↓                  ______
     ↓ ----------->    |情况 B|
                    　　———————
    4. 为了完成全部的建立 ddp 所需要的操作，三种情况都需要做的事情，以及每件事情的职责归属：

                                   情况 A          ｜          情况 B           ｜          情况 C
  ________________________________________________________________________________________________________
  配置 ddp 所      ｜ TorchDDPDriver.open_subprocess ｜ torch.distributed.launch｜ torch.distributed.launch
  需要的环境变量    ｜                                ｜                         ｜
  ————————————————————————————————————————————————————————————————————————————————————————————————————————
  开启多个进程     ｜ TorchDDPDriver.open_subprocess ｜ torch.distributed.launch｜ torch.distributed.launch
  ————————————————————————————————————————————————————————————————————————————————————————————————————————
  调用 dist.      ｜                                ｜                          ｜
  init_process\  ｜      TorchDDPDriver.setup      ｜    TorchDDPDriver.setup  ｜         用户自己调用
  _group 函数     ｜                                ｜                          ｜
  ————————————————————————————————————————————————————————————————————————————————————————————————————————
  设置 TorchDDPDriver ｜                            ｜                          ｜
  的 world_size 和    ｜    TorchDDPDriver.setup    ｜  TorchDDPDriver.setup    ｜   TorchDDPDriver.setup
  global_rank 属性    ｜                            ｜                          ｜
  ————————————————————————————————————————————————————————————————————————————————————————————————————————

Part 3：其它的处理细节：
    1. 环境变量；
    fastNLP 的 `TorchDDPDriver` 运行时所需要的环境变量分为两种，一种是 torch 的 ddp 运行所需要的环境变量；另一种是 fastNLP 自己
     的环境变量。前者的配置情况如上表所示；而后者中的大多数环境变量则是在用户 import fastNLP 时就设置好了；
    2. parallel_device, model_device 和 data_device 的关系；
    parallel_device 为 `TorchDDPDriver` 的参数，model_device 和 data_device 都为 driver 的属性；
    其中 data_device 仅当情况 C 时由用户自己指定；如果其不为 None，那么在模型 forward 的时候，我们就会将数据迁移到 data_device 上；
    model_device 永远都为单独的一个 torch.device；

                                   情况 A          ｜          情况 B           ｜          情况 C
  ________________________________________________________________________________________________________
  parallel_device ｜   由用户传入trainer的参数        ｜  为 torch.device(        ｜     为 torch.device(
                  ｜  device 决定，必须是一个list，   ｜   "cuda:{local_rank}")   ｜    "cuda:{local_rank}")
                  ｜  其中每一个对象都是 torch.device ｜                          ｜
  ————————————————————————————————————————————————————————————————————————————————————————————————————————
  model_device    ｜ parallel_device[local_rank]   ｜      parallel_device     ｜            None
  ————————————————————————————————————————————————————————————————————————————————————————————————————————
  data_device     ｜         model_device          ｜       model_device       ｜  由用户传入 trainer 的参数
                  ｜                               ｜                          ｜     data_device 决定
  ————————————————————————————————————————————————————————————————————————————————————————————————————————

    3. _DDPWrappingModel 的作用；
    因为我们即需要调用模型的 `train_step`、`evaluate_step`、`test_step` 方法，又需要通过 `DistributedDataParallel` 的
     forward 函数来帮助我们同步各个设备上的梯度，因此我们需要先将模型单独包裹一层，然后在 forward 的时候，其先经过 `DistributedDataParallel`
     的 forward 方法，然后再经过 `_DDPWrappingModel` 的 forward 方法，我们会在该 forward 函数中进行判断，确定调用的是模型自己的
     forward 函数，还是 `train_step`、`evaluate_step`、`test_step` 方法。

    4. 当某一个进程出现 exception 后，`TorchDDPDriver` 的处理；

    不管是什么情况，`TorchDDPDriver` 在 `setup` 函数的最后，都会将所有进程的 pid 主动记录下来，这样当一个进程出现 exception 后，
     driver 的 on_exception 函数就会被 trainer 调用，其会调用 os.kill 指令将其它进程 kill 掉；
"""

import os
import sys
import __main__
import socket
import numpy as np
from time import sleep
from typing import List, Optional, Union, Dict, Tuple, Callable

from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import BatchSampler

__all__ = [
    'TorchDDPDriver'
]

from .torch_driver import TorchDriver
from fastNLP.core.drivers.torch_driver.utils import (
    _DDPWrappingModel,
    replace_sampler,
    replace_batch_sampler
)
from fastNLP.core.drivers.utils import distributed_open_proc
from fastNLP.core.utils import auto_param_call, check_user_specific_params
from fastNLP.core.samplers import ReproducibleSampler, RandomSampler, UnrepeatedSequentialSampler, \
    ReproducibleBatchSampler, \
    re_instantiate_sampler, UnrepeatedSampler, conversion_between_reproducible_and_unrepeated_sampler
from fastNLP.envs import FASTNLP_DISTRIBUTED_CHECK, FASTNLP_GLOBAL_RANK, FASTNLP_GLOBAL_SEED, FASTNLP_NO_SYNC
from fastNLP.core.log import logger
from fastNLP.core.drivers.torch_driver.dist_utils import fastnlp_torch_all_gather, fastnlp_torch_broadcast_object, \
    convert_sync_batchnorm
from .utils import _check_dataloader_args_for_distributed


class TorchDDPDriver(TorchDriver):
    r"""
    ``TorchDDPDriver`` 通过开启多个进程，让每个进程单独使用一个 gpu 设备来实现分布式训练。

    .. note::

        您在绝大多数情况下不需要自己使用到该类，通过向 ``Trainer`` 传入正确的参数，您可以方便快速地部署您的分布式训练。

        ``TorchDDPDriver`` 目前支持的三种启动方式：

            1. 用户自己不进行 ``ddp`` 的任何操作，直接使用我们的 ``Trainer``，这时是由我们自己使用 ``open_subprocesses`` 拉起多个进程，
            然后 ``TorchDDPDriver`` 自己通过调用 ``dist.init_process_group`` 来初始化 ddp 的通信组；（情况 A）

            .. code-block::

                trainer = Trainer(
                    ...
                    driver='torch',
                    device=[0, 1]
                )
                trainer.run()

            通过运行 ``python train.py`` 启动；

            2. 用户同样不在 ``Trainer`` 之外初始化 ``ddp``，但是用户自己使用 ``python -m torch.distributed.launch`` 拉起来创建多个进程，这时我们仍旧
            会通过调用 ``dist.init_process_group`` 来初始化 ``ddp`` 的通信组；（情况 B）

            .. code-block::

                trainer = Trainer(
                    ...
                    driver='torch',
                    device=None, # fastNLP 会忽略传入的 device，并根据 local_rank 自动分配
                )
                trainer.run()

            通过运行 ``python -m torch.distributed.launch --nproc_per_node 2 train.py`` 启动；

            3. 用户自己在外面初始化 ``DDP``，并且通过 ``python -m torch.distributed.launch`` 拉起，这时无论是多个进程的拉起和 ddp 的通信组的建立
            都由用户自己操作，我们只会在 ``driver.setup`` 的时候对 ``TorchDDPDriver`` 设置一些必要的属性值；（情况 C）

            .. code-block::

                import torch.distributed as dist
                from torch.nn.parallel import DistributedDataParallel

                # 获取当前的进程信息；
                ...

                # 初始化 ddp 不同进程间的通信组；
                dist.init_process_group(...)

                # 初始化模型使用 DistributedDataParallel 包裹；
                model = Model()
                model = DistributedDataParallel(model, ...)

                # 注意此时仍旧不需要您主动地将 datalaoder 的 sampler 替换为 DistributedSampler；
                trainer = Trainer(
                    ...
                    driver='torch',
                    device=None, # fastNLP 会忽略传入的 device，并根据 local_rank 自动分配
                )
                trainer.run()

            通过运行 ``python -m torch.distributed.launch --nproc_per_node 2 train.py`` 启动；

        注意多机的启动强制要求用户在每一台机器上使用 ``python -m torch.distributed.launch`` 启动；因此我们不会在 ``TorchDDPDriver`` 中保存
        任何当前有多少台机器的信息。

    :param model: 传入给 ``Trainer`` 的 ``model`` 参数
    :param parallel_device: 用于分布式训练的 ``gpu`` 设备
    :param is_pull_by_torch_run: 标志当前的脚本的启动是否由 ``python -m torch.distributed.launch`` 启动的
    :param fp16: 是否开启 fp16 训练
    :param torch_kwargs: 
        * *ddp_kwargs* -- 用于在使用 ``TorchDDPDriver`` 时指定 ``DistributedDataParallel`` 初始化时的参数；例如传入
          ``{'find_unused_parameters': True}`` 来解决有参数不参与前向运算导致的报错等
        * *set_grad_to_none* -- 是否在训练过程中在每一次 optimizer 更新后将 grad 置为 ``None``
        * *non_blocking* -- 表示用于 :meth:`torch.Tensor.to` 方法的参数 non_blocking
        * *gradscaler_kwargs* -- 用于 ``fp16=True`` 时，提供给 :class:`torch.amp.cuda.GradScaler` 的参数
    :kwargs:
        * *wo_auto_param_call* (``bool``) -- 是否关闭在训练时调用我们的 ``auto_param_call`` 函数来自动匹配 batch 和前向函数的参数的行为
        * *sync_bn* (``bool``) -- 是否要将模型中可能存在的 BatchNorm 层转换为可同步所有卡数据计算均值和方差的 SyncBatchNorm 层，默认为 True

        .. note::

            关于该参数的详细说明，请参见 :class:`~fastNLP.core.controllers.Trainer` 中的描述；函数 ``auto_param_call`` 详见 :func:`fastNLP.core.utils.auto_param_call`。
    """

    def __init__(
            self,
            model,
            parallel_device: Optional[Union[List["torch.device"], "torch.device"]],
            is_pull_by_torch_run: bool = False,
            fp16: bool = False,
            torch_kwargs: Dict = None,
            **kwargs
    ):

        # 在加入很多东西后，需要注意这里调用 super 函数的位置；
        super(TorchDDPDriver, self).__init__(model, fp16=fp16, torch_kwargs=torch_kwargs, **kwargs)

        if isinstance(model, torch.nn.DataParallel):
            raise ValueError(f"Parameter `model` can not be `DataParallel` in `TorchDDPDriver`, it should be "
                             f"`torch.nn.Module` or `torch.nn.parallel.DistributedDataParallel` type.")

        # 如果用户自己在外面初始化 DDP，那么其一定是通过 python -m torch.distributed.launch 拉起的；
        self.is_pull_by_torch_run = is_pull_by_torch_run
        self.parallel_device = parallel_device
        if not is_pull_by_torch_run and parallel_device is None:
            raise ValueError(
                "Parameter `parallel_device` can not be None when using `TorchDDPDriver`. This error is caused "
                "when your value of parameter `device` is `None` in your `Trainer` instance.")

        # 注意我们在 initialize_torch_driver 中的逻辑就是如果是 is_pull_by_torch_run，那么我们就直接把 parallel_device 置为当前进程的gpu；
        if is_pull_by_torch_run:
            self.model_device = parallel_device
        else:
            # 我们的 model_device 一定是 torch.device，而不是一个 list；
            self.model_device = parallel_device[self.local_rank]

        # 如果用户自己在外面初始化了 DDP；
        self.outside_ddp = False
        if dist.is_initialized() and FASTNLP_DISTRIBUTED_CHECK not in os.environ and \
                "fastnlp_torch_launch_not_ddp" not in os.environ:
            # 如果用户自己在外面初始化了 DDP，那么我们要求用户传入的模型一定是已经由 DistributedDataParallel 包裹后的模型；
            if not isinstance(model, DistributedDataParallel):
                raise RuntimeError(
                    "It is not allowed to input a normal model instead of `DistributedDataParallel` when"
                    "you initialize the ddp process out of our control.")

            self.outside_ddp = True
            # 用户只有将模型上传到对应机器上后才能用 DistributedDataParallel 包裹，因此如果用户在外面初始化了 DDP，那么在 TorchDDPDriver 中
            #  我们就直接将 model_device 置为 None；
            self.model_device = None

        # 当用户自己在外面初始化 DDP 时我们会将 model_device 置为 None，这是用户可以通过 `data_device` 将对应的数据移到指定的机器上;
        self._data_device = kwargs.get("data_device", None)
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

        self._fsdp_kwargs = self._torch_kwargs.get("ddp_kwargs", {})
        check_user_specific_params(self._fsdp_kwargs, DistributedDataParallel.__init__, DistributedDataParallel.__name__)
        if len(self.model._buffers) != 0 and self._fsdp_kwargs.get("broadcast_buffers", None) is None:
            logger.info("Notice your model has buffers and you are using `TorchDDPDriver`, but you do not set "
                        "'broadcast_buffers' in your trainer. Cause in most situations, this parameter can be set"
                        " to 'False' to avoid redundant data communication between different processes.")

        self.output_from_new_proc = kwargs.get("output_from_new_proc", "only_error")
        assert isinstance(self.output_from_new_proc, str), "Parameter `output_from_new_proc` can only be `str` type."
        if self.output_from_new_proc not in {"all", "ignore", "only_error"}:
            os.makedirs(name=self.output_from_new_proc, exist_ok=True)
            self.output_from_new_proc = os.path.abspath(self.output_from_new_proc)

        self._has_setup = False  # 设置这一参数是因为 evaluator 中也会进行 setup 操作，但是显然是不需要的也不应该的；
        self._has_ddpwrapped = False  # 判断传入的模型是否经过 _has_ddpwrapped 包裹；
        # sync_bn 表示是否要将模型中可能存在的 BatchNorm 层转换为可同步所有卡数据计算均值和方差的 SyncBatchNorm 层；
        self.sync_bn = kwargs.get("sync_bn", True)

    def setup(self):
        r"""
        准备分布式环境，该函数主要做以下两件事情：

            1. 开启多进程，每个 ``gpu`` 设备对应单独的一个进程；
            2. 每个进程将模型迁移到自己对应的 ``gpu`` 设备上；然后使用 ``DistributedDataParallel`` 包裹模型；
        """
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
                logger.info(f"World size: {self.world_size}, Global rank: {self.global_rank}")

                if not dist.is_initialized():
                    dist.init_process_group(
                        backend="nccl", rank=self.global_rank, world_size=self.world_size
                    )

                os.environ["fastnlp_torch_launch_not_ddp"] = "yes"

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
                dist.init_process_group(
                    backend="nccl", rank=self.global_rank, world_size=self.world_size
                )
            # 用户在这个 trainer 前面又初始化了一个 trainer，并且使用的是 TorchDDPDriver；
            else:
                # 如果 `dist.is_initialized() == True`，那么说明 TorchDDPDriver 在之前已经初始化并且已经 setup 过一次，那么我们需要保证现在
                #  使用的（即之后的）TorchDDPDriver 的设置和第一个 TorchDDPDriver 是完全一样的；
                pre_num_processes = int(os.environ[FASTNLP_DISTRIBUTED_CHECK])
                if pre_num_processes != len(self.parallel_device):
                    raise RuntimeError(
                        "Notice you are using `TorchDDPDriver` after one instantiated `TorchDDPDriver`, it is not"
                        "allowed that your second `TorchDDPDriver` has a new setting of parameters "
                        "`num_nodes` and `num_processes`.")
                self.world_size = dist.get_world_size()
                self.global_rank = dist.get_rank()

        if not self.outside_ddp:
            self.configure_ddp()

        if self.sync_bn:
            self.model = convert_sync_batchnorm(self.model)

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
        self._pids = self._pids[node_rank * local_world_size: (node_rank + 1) * local_world_size]
        self._pids = self.tensor_to_numeric(self._pids)

    def configure_ddp(self):
        torch.cuda.set_device(self.model_device)
        self.model.to(self.model_device)
        if not isinstance(self.model, DistributedDataParallel):
            self.model = DistributedDataParallel(
                # 注意这里的 self.model_device 是 `torch.device` type，因此 self.model_device.index；
                _DDPWrappingModel(self.model), device_ids=[self.model_device.index],
                **self._fsdp_kwargs
            )
            self._has_ddpwrapped = True

    def open_subprocess(self):
        if self.local_rank == 0:
            # Script called as `python a/b/c.py`
            if __main__.__spec__ is None:  # pragma: no-cover
                # pull out the commands used to run the script and resolve the abs file path
                command = sys.argv
                command[0] = os.path.abspath(command[0])
                # use the same python interpreter and actually running
                command = [sys.executable] + command
            # Script called as `python -m a.b.c`
            else:
                command = [sys.executable, "-m", __main__.__spec__.name] + sys.argv[1:]

            os.environ['MASTER_ADDR'] = self.master_address
            os.environ['MASTER_PORT'] = self.master_port

            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = str(self.local_rank)
            os.environ["WORLD_SIZE"] = f"{self.world_size}"

            os.environ[FASTNLP_DISTRIBUTED_CHECK] = f"{len(self.parallel_device)}"
            os.environ[FASTNLP_GLOBAL_RANK] = "0"
            logger._set_distributed()

            interactive_ddp_procs = []

            for rank in range(1, len(self.parallel_device)):
                env_copy = os.environ.copy()
                env_copy["LOCAL_RANK"] = f"{rank}"
                env_copy["RANK"] = f"{rank}"

                # 如果是多机，一定需要用户自己拉起，因此我们自己使用 open_subprocesses 开启的进程的 FASTNLP_GLOBAL_RANK 一定是 LOCAL_RANK；
                env_copy[FASTNLP_GLOBAL_RANK] = str(rank)

                proc = distributed_open_proc(self.output_from_new_proc, command, env_copy, self.global_rank)

                interactive_ddp_procs.append(proc)
                delay = np.random.uniform(1, 5, 1)[0]
                sleep(delay)

    @property
    def master_address(self) -> str:
        """
        分布式训练中的地址 ``MASTER_ADDR``
        """
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    @property
    def master_port(self) -> str:
        """
        分布式训练使用的端口 ``MASTER_PORT``
        """
        if self.outside_ddp:
            return os.environ.get("MASTER_PORT")
        if self._master_port is None:
            self._master_port = os.environ.get("MASTER_PORT", find_free_network_port())
        return self._master_port

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
    def local_rank(self) -> int:  # 这个不会受到 all_rank_call_context 的影响
        """
        当前进程的局部编号 ``local_rank``
        """
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def data_device(self):
        if self.outside_ddp:
            return self._data_device
        return self.model_device

    def model_call(self, batch, fn: Callable, signature_fn: Optional[Callable]) -> Dict:
        if self._has_ddpwrapped:
            return self.model(batch, fastnlp_fn=fn, fastnlp_signature_fn=signature_fn,
                              wo_auto_param_call=self.wo_auto_param_call)
        else:
            if isinstance(batch, Dict) and not self.wo_auto_param_call:
                return auto_param_call(fn, batch, signature_fn=signature_fn)
            else:
                return fn(batch)

    def get_model_call_fn(self, fn: str) -> Tuple:
        model = self.unwrap_model()
        if self._has_ddpwrapped:
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
                logger.warning("Notice your model is a `DistributedDataParallel` model. And your model also implements "
                               f"the `{fn}` method, which we can not call actually, we will"
                               " call `forward` function instead of `train_step` and you should note that.")
            elif fn not in {"train_step", "evaluate_step"}:
                raise RuntimeError(f"There is no `{fn}` method in your model. And also notice that your model is a "
                                   "`DistributedDataParallel` model, which means that we will only call model.forward "
                                   "function when we are in forward propagation.")

            return self.model, model.forward

    def set_dist_repro_dataloader(self, dataloader,
                                  dist: Optional[Union[str, ReproducibleSampler, ReproducibleBatchSampler]] = None,
                                  reproducible: bool = False):
        # 如果 dist 为 ReproducibleBatchSampler, ReproducibleSampler 说明是在断点重训时 driver.load_checkpoint 函数调用；
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
            raise ValueError(
                "Parameter `dist_sampler` can only be one of three values: ('dist', 'unrepeatdist', None).")

    def is_global_zero(self):
        r"""
        :return: 当前的进程是否在全局上是进程 0 。
        """
        return self.global_rank == 0

    def get_model_no_sync_context(self):
        r"""
        :return: 一个 ``context`` 上下文环境，用于关闭各个进程之间的同步。
        """
        # 注意此时的 model 是 "DistributedDataParallel" 对象；
        return self.model.no_sync

    def unwrap_model(self):
        r"""
        :return: 没有经过 ``DistributedDataParallel`` 包裹的原始模型。
        """
        _module = self.model.module
        if isinstance(_module, _DDPWrappingModel):
            return _module.model
        else:
            return _module

    def get_local_rank(self) -> int:
        r"""
        :return: 当前进程局部的进程编号。
        """
        return self.local_rank

    def barrier(self):
        r"""
        通过使用该函数来使得各个进程之间同步操作。
        """
        if int(os.environ.get(FASTNLP_NO_SYNC, 0)) < 1 and dist.is_initialized():  # 当 FASTNLP_NO_SYNC 小于 1 时实际执行
            torch.distributed.barrier(async_op=False)

    def is_distributed(self):
        r"""
        :return: 当前使用的 driver 是否是分布式的 driver，对于 ``TorchDDPDriver`` 来说，该函数一定返回 ``True``。
        """
        return True

    def broadcast_object(self, obj, src: int = 0, group=None, **kwargs):
        r"""
        从 ``src`` 端将 ``obj`` 对象（可能是 tensor ，可能是 object ）广播到其它进程。如果是非 tensor 的对象会尝试使用 pickle 进行打包进行
        传输，然后在接收处处再加载回来。仅在分布式的 driver 中有实际意义。

        :param obj: obj，可能是 Tensor 或 嵌套类型的数据
        :param src: 发送方的 ``global_rank``
        :param group: 进程所在的通信组
        :return: 如果当前 rank 是接收端，则返回接收到的参数；如果是 source 端则返回发送的内容。如果环境变量 ``FASTNLP_NO_SYNC`` 为 **2** 则
            返回 ``None``
        """
        if int(os.environ.get(FASTNLP_NO_SYNC, 0)) == 2:  # 如果 FASTNLP_NO_SYNC == 2 直接返回。
            return
        return fastnlp_torch_broadcast_object(obj, src, device=self.data_device, group=group)

    def all_gather(self, obj, group) -> List:
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
        if int(os.environ.get(FASTNLP_NO_SYNC, 0)) == 2:  # 如果 FASTNLP_NO_SYNC 表示不执行
            return [obj]
        return fastnlp_torch_all_gather(obj, group=group)

    def on_exception(self):
        super().on_exception()
        dist.destroy_process_group()  # 防止在之后的 barrier 出现卡死的问题。


def find_free_network_port() -> str:
    """
    在 localhost 上找到一个空闲端口；
    当我们不想连接到真正的主节点但必须设置“MASTER_PORT”环境变量时在单节点训练中很有用。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return str(port)
