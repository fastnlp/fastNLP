import os
from typing import List, Union, Optional, Dict, Tuple, Callable

from .paddle_driver import PaddleDriver
from .fleet_launcher import FleetLauncher
from .utils import (
    _FleetWrappingModel, 
    reset_seed,
    replace_sampler,
    replace_batch_sampler,
)
from .dist_utils import fastnlp_paddle_all_gather, fastnlp_paddle_broadcast_object

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.core.utils import (
    auto_param_call,
    check_user_specific_params,
    is_in_paddle_dist,
    rank_zero_rm
)
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
from fastNLP.envs.env import FASTNLP_DISTRIBUTED_CHECK, FASTNLP_GLOBAL_SEED, FASTNLP_NO_SYNC
from fastNLP.core.log import logger

if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle import DataParallel
    import paddle.distributed.fleet as fleet
    import paddle.distributed as paddledist
    from paddle.optimizer import Optimizer
    from paddle.fluid.reader import _DatasetKind
    from paddle.fluid.dygraph import parallel_helper

__all__ = [
    "PaddleFleetDriver",
]

class PaddleFleetDriver(PaddleDriver):
    def __init__(
            self, 
            model, 
            parallel_device: Optional[Union[List[int], int]],
            is_pull_by_paddle_run: bool = False,
            fp16: bool = False,
            **kwargs
    ):
        r"""
        通过使用 PaddlePaddle 的 Fleet 框架启动多卡进程的 Driver。
        需要注意的一点是，由于 PaddlePaddle 框架的特性，如果直接使用在 rank0 拉起其它进程的方法的话，如果不加以任何限制，PaddlePaddle会出现
        第一次前向传播后卡住或占用所有显卡的现象；为了解决这一问题，我们在引入 FastNLP 时，会使用 `CUDA_VISIBLE_DEVICES` 将设备限制在卡0上，
        而用户如果使用了这一环境变量，我们会将其储存在 `USER_CUDA_VISIBLE_DEVICES` 中，并且通过一定的手段实现了转换（详细的设置请参见：
        `fastNLP/envs/set_backend.py`）。在拉起其它进程的时候，我们会如法炮制，将环境限制在对应的设备上。

        `PaddleFleetDriver` 目前支持的三种启动方式：
        1. 用户自己不进行分布式的任何操作，直接使用我们的 Trainer，这时是由我们自己使用 `FleetLauncher` 拉起多个进程，
         然后 `PaddleFleetDriver` 自己通过调用 `fleet.init` 来初始化 ddp 的通信组；（情况 A）
        2. 用户同样不在 Trainer 之外初始化分布式训练，但是用户自己使用 python -m paddle.distributed.launch 拉起来创建多个进程，这时我们仍旧
         会通过调用 `fleet.init` 来初始化 ddp 的通信组；（情况 B）
        3. 用户自己在外面初始化分布式，并且通过 python -m paddle.distributed.launch 拉起，这时无论是多个进程的拉起和通信组的建立
         都由用户自己操作，我们只会在 driver.setup 的时候对 `PaddleFleetDriver` 设置一些必要的属性值；（情况 C）

        注意多机的启动强制要求用户在每一台机器上使用 python -m paddle.distributed.launch 启动；因此我们不会在 `PaddleFleetDriver` 中保存
         任何当前有多少台机器的信息；

        Part 1：三种启动方式的具体分析：
        （1）对于用户运行的脚本中，如果 `driver.setup` 只会被调用一次（意味着用户的启动脚本中只初始化了一个 trainer/evaluator）时，
         `PaddleFleetDriver` 在初始化以及 `setup` 函数中会做的事情分别如下所示：
            -> 情况 A：这种情况下用户传入的 model 在一定是普通的 model（没有经 `DataParallel` 包裹的model），
             因为 `Parallel` 的使用一定要求 fleet.init 已经被调用用来建立当前的 ddp 通信组；但是这意味着如果
             用户需要使用 2 张以上的显卡，那么其必然需要使用 paddle.distributed.launch 来启动，意味着就不是情况 A 了；
             这时我们首先会调用 `FleetLauncher.launch` 函数来拉起多个进程，其中进程的数量等于用户传入给 trainer 的使用的 gpu
             的数量（例如 `Trainer` 中的参数是 device=[0, 1, 6, 7]，那么我们就会使用第 0、1、6、7 张 gpu 来拉起 4 个进程）；
             接着我们会调用 `fleet.init` 来初始化各个进程之间的通信组；
             这里需要注意拉起的新的进程会从前到后完整地运行一遍用户的启动脚本（例如 main.py），因此也都会运行这两个函数，但是需要注意只有进程 0
             才会去真正地运行 `FleetLauncher.launch`；进程 0 运行到 `fleet.init`，paddle 会阻塞进程 0 继续
             向前运行，直到其它进程也运行到这里；
             最后我们会设置这个进程对应的 device，然后将模型迁移到对应的机器上，再使用 `DataParallel` 将模型包裹；
             至此，paddle 分布式的环境配置过程全部完成；

            -> 情况 B：注意这种情况我们直接限定了用户是通过 paddle.distributed.launch 拉起，并且没有自己建立分布式的通信组。这时在
             `PaddleFleetDriver` 的初始化和 setup 函数的调用过程中，与情况 A 首要的不同就在于用户在 trainer 中输入的参数 device 不再有效，
             这时每个进程所使用的 gpu 是我们直接通过 `CUDA_VISIBLE_DEVICE` 来配置的；因此，如果用户想要实现使用特定 gpu
             设备的目的，可以通过自己设置环境变量实现（例如 os.environ["CUDA_VISIBLE_DEVICE"] 来实现，我们会通过一定的手段将其保存起来）；
             剩下的操作和情况 A 类似；

            -> 情况 C：注意这种情况我们限定了用户是通过 paddle.distributed.launch 拉起，并且 ddp 的通信组也是由自己建立。这时基本上所有的
             与操作相关的操作都应当由用户自己完成，包括迁移模型到对应 gpu 上以及将模型用 `DataParallel` 包裹等。
        （2）如果 `driver.setup` 函数在脚本中会被调用两次及以上（意味着用户的启动脚本初始化了两个及以上的 trainer/evaluator）时：
        注意这种情况下我们是会保证前后两个 trainer/evaluator 使用的 `PaddleFleetDriver` 以及其初始化方式的一致性，换句话说，如果 trainer1
         检测到的启动方式是 '情况 A'，那么我们会保证 trainer2 检测到的启动方式同样是 '情况A'（即使这需要一些额外的处理）；因此这里我们主要讨论
         我们是通过怎样的操作来保证 trainer2/3/... 检测到的启动方式是和 trainer1 一致的；简单来说，我们是通过使用环境变量来标记每一种不同的
         启动方式来实现这一点的：
        我们会使用 `FASTNLP_DISTRIBUTED_CHECK` 来标记 '情况 A'，使用 `fastnlp_torch_launch_not_ddp` 来标记 '情况 B'，意味着我们在
         使用 '情况 A' 来启动 `PaddleFleetDriver` 时，我们会将 `FASTNLP_DISTRIBUTED_CHECK` 这一字符串注入到环境变量中，而 '情况 B' 时则
         会将 `fastnlp_torch_launch_not_ddp` 这一字符串注入到环境变量中。因此在 trainer2 的 `PaddleFleetDriver` 的初始化和 setup 过程中，
         如果检测到这些特殊的环境变量，我们就会将启动方式变更为其对应的启动方式，即使其它的参数特征属于另外的启动方式。

        Part 2：对应的代码细节：
            1. 如何判断当前的各进程之间的通信组已经被建立（fleet 已经被初始化）；
                parallel_helper._is_parallel_ctx_initialized()；
            2. 如何判断不同的进程是否是由 `python -m paddle.distributed.launch` 拉起还是由我们的 `FleetLauncher.launch()`
             函数拉起；
                我们会在用户脚本 `import fastNLP` 的时候检测当前的环境变量中是否有 'PADDLE_RANK_IN_NODE'、'PADDLE_TRAINER_ID' 
                以及没有 `FASTNLP_DISTRIBUTED_CHECK`，
                如果满足条件，则我们会向环境变量中注入特殊的值 'FASTNLP_BACKEND_LAUNCH' 来标记用户是否使用了 `python -m paddle.distributed.launch`
                来拉起多个进程；
            3. 整体的处理判断流程：
                 ___________________________________
                ｜进入 PaddleFleetDriver 的 __init__ 函数｜
                 ———————————————————————————————————
                                 ↓
            ___________________________________________________
           ｜ 判断不同的进程是否是由 paddle.distributed.launch 拉起 ｜
           ｜（或者我们自己的 FleetLauncher 函数拉起）           ｜  -------------->
            ———————————————————————————————————————————————————                　｜
                                 ↓ 是由 paddle.distributed.launch 拉起            ｜ 我们自己的 FleetLauncher 函数拉起多个进程
                    　_____________________________           　　　　　　　　　　　｜　
             ←←←←←  ｜ 检测用户是否自己初始化了 fleet ｜       　　　　　　　　　　　　　｜
             ↓       —————————————————————————————　　　　　　　　　　　　　　　　　 ↓
             ↓                   ↓ 是                                         ________
             ↓                  ______                                      ｜ 情况 A ｜
             ↓ 否               |情况 C|                                      —————————
             ↓                 ———————
             ↓
             ↓                  ______
             ↓ ----------->    |情况 B|
                            　　———————
            4. 为了完成全部的建立分布式所需要的操作，三种情况都需要做的事情，以及每件事情的职责归属：

                                           情况 A          ｜          情况 B           ｜          情况 C
          ________________________________________________________________________________________________________
          配置 fleet 所    ｜     FleetLauncher.launch     ｜ paddle.distributed.launch｜ paddle.distributed.launch
          需要的环境变量    ｜                              ｜                         ｜
          ————————————————————————————————————————————————————————————————————————————————————————————————————————
          开启多个进程     ｜     FleetLauncher.launch     ｜ paddle.distributed.launch｜ paddle.distributed.launch
          ————————————————————————————————————————————————————————————————————————————————————————————————————————
          调用 fleet.init函数 ｜  PaddleFleetDriver.setup    ｜ PaddleFleetDriver.setup  ｜        用户自己调用
          ————————————————————————————————————————————————————————————————————————————————————————————————————————
          设置 PaddleFleetDriver ｜                            ｜                          ｜
          的 world_size 和       ｜  PaddleFleetDriver.setup   ｜ PaddleFleetDriver.setup  ｜   PaddleFleetDriver.setup
          global_rank 属性       ｜                            ｜                          ｜
          ————————————————————————————————————————————————————————————————————————————————————————————————————————

        Part 3：其它的处理细节：
            1. 环境变量；
            fastNLP 的 `PaddleFleetDriver` 运行时所需要的环境变量分为两种，一种是 paddle fleet 运行所需要的环境变量；另一种是 fastNLP 自己
             的环境变量。前者的配置情况如上表所示；而后者中的大多数环境变量则是在用户 import fastNLP 时就设置好了；
            2. parallel_device, model_device 和 data_device 的关系；
            parallel_device 为 `PaddleFleetDriver` 的参数，model_device 和 data_device 都为 driver 的属性；
            其中 data_device 仅当情况 C 时由用户自己指定；如果其不为 None，那么在模型 forward 的时候，我们就会将数据迁移到 data_device 上；
            model_device 永远都为单独的一个 torch.device；

                                           情况 A          ｜          情况 B           ｜          情况 C
          ________________________________________________________________________________________________________
          parallel_device ｜   由用户传入trainer的参数        ｜                         ｜ 
                          ｜  device 决定，必须是一个list，   ｜ 为 CUDA_VISIBLE_DEVICES ｜ 为 CUDA_VISIBLE_DEVICES
                          ｜  其中每一个对象都是 int          ｜                          ｜
          ————————————————————————————————————————————————————————————————————————————————————————————————————————
          model_device    ｜ parallel_device[local_rank]   ｜      parallel_device     ｜            None
          ————————————————————————————————————————————————————————————————————————————————————————————————————————
          data_device     ｜         model_device          ｜       model_device       ｜  由用户传入 trainer 的参数
                          ｜                               ｜                          ｜     data_device 决定
          ————————————————————————————————————————————————————————————————————————————————————————————————————————

            3. _DDPWrappingModel 的作用；
            因为我们即需要调用模型的 `train_step`、`evaluate_step`、`test_step` 方法，又需要通过 `DataParallel` 的forward 函数来帮助
            我们同步各个设备上的梯度，因此我们需要先将模型单独包裹一层，然后在 forward 的时候，其先经过 `DataParallel` 的 forward 方法，
            然后再经过 `_DDPWrappingModel` 的 forward 方法，我们会在该 forward 函数中进行判断，确定调用的是模型自己的 forward 函数，还是
            `train_step`、`evaluate_step`、`test_step` 方法。

            4. 当某一个进程出现 exception 后，`PaddleFleetDriver` 的处理；

            不管是什么情况，`PaddleFleetDriver` 在 `setup` 函数的最后，都会将所有进程的 pid 主动记录下来，这样当一个进程出现 exception 后，
             driver 的 on_exception 函数就会被 trainer 调用，其会调用 os.kill 指令将其它进程 kill 掉；
        """
        super(PaddleFleetDriver, self).__init__(model, fp16=fp16, **kwargs)

        # 如果不是通过 launch 启动，要求用户必须传入 parallel_device
        if not is_pull_by_paddle_run and parallel_device is None:
            raise ValueError("Parameter `parallel_device` can not be None when using `PaddleFleetDriver`. This error is caused "
                             "when your value of parameter `device` is `None` in your `Trainer` instance.")
        
        # 如果用户自己初始化了 paddle 的分布式训练那么一定是通过 launch 拉起的
        # 这个参数会在 initialize_paddle_drvier 中设置。
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
            # 我们就直接将 model_device 置为 None；
            self._model_device = None

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
        self.gloo_rendezvous_dir = None

        # 分布式环境的其它参数设置
        self._fleet_kwargs = kwargs.get("paddle_fleet_kwargs", {})
        check_user_specific_params(self._fleet_kwargs, DataParallel.__init__)
        # fleet.init 中对于分布式策略的设置，详情可以参考 PaddlePaddle 的官方文档
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

        self._has_setup = False # 设置这一参数是因为 evaluator 中也会进行 setup 操作，但是显然是不需要的也不应该的；
        self._has_fleetwrapped = False  # 判断传入的模型是否经过 _has_fleetwrapped 包裹；

    def setup(self):
        """
        根据不同的情况进行不同的设置。
        1、如果是通过 paddle.distributed.launch 方法启动时，则根据已经设置好的环境获取
           分布式的属性。
        2、否则，调用 FleetLauncher 类启动子进程
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
                # 拉起子进程并设置相应的属性
                self.init_fleet_and_set()
            # 用户在这个 trainer 前面又初始化了一个 trainer，并且使用的是 PaddleFleetDriver；
            else:
                # 已经设置过一次，保证参数必须是一样的
                pre_gpus = os.environ[FASTNLP_DISTRIBUTED_CHECK]
                pre_gpus = [int (x) for x in pre_gpus.split(",")]
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

    def init_fleet_and_set(self):
        """
        使用 FleetLauncher 拉起子进程
        """
        if self.local_rank == 0:
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

    def set_from_fleet_environment(self):
        """
        当用户使用了 `python -m paddle.distributed.launch xxx.py` 启动时，我们需要
        根据 paddle 设置的环境变量来获得各种属性
        """
        self.world_size = paddledist.get_world_size()
        self.global_rank = paddledist.get_rank()

    def barrier(self):
        r"""
        用于在多进程工作时同步各进程的工作进度，运行快的进程运行到这里会等待运行慢的进程，只有所有进程都运行到此函数时，所有的进程才会继续运行；
        仅在多分布式训练场景中有使用。

        注意，该函数的行为会受到 FASTNLP_NO_SYNC 的影响。仅当 FASTNLP_NO_SYNC 在 os.environ 中不存在，或小于 1 时才真的执行 barrier 。
        """
        if int(os.environ.get(FASTNLP_NO_SYNC, 0)) < 1:  # 当 FASTNLP_NO_SYNC 小于 1 时实际执行
            paddledist.barrier()

    def configure_fleet(self):
        """
        将模型用 DataParallel 和自定义的类型包裹起来
        """
        if not self._has_fleetwrapped and not isinstance(self.model, DataParallel):
            self.model = DataParallel(
                _FleetWrappingModel(self.model),
                **self._fleet_kwargs
            )
            self._has_fleetwrapped = True

    def on_exception(self):
        """
        该函数用于在训练或者预测过程中出现错误时正确地关掉其它的进程，这一点是通过在多进程 driver 调用 open_subprocess 的时候将每一个进程
         的 pid 记录下来，然后在出现错误后，由出现错误的进程手动地将其它进程 kill 掉；

        因此，每一个多进程 driver 如果想要该函数能够正确地执行，其需要在自己的 open_subprocess（开启多进程的函数）中正确地记录每一个进程的
         pid 的信息；
        """
        rank_zero_rm(self.gloo_rendezvous_dir)
        super().on_exception()

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

    def model_call(self, batch, fn: Callable, signature_fn: Optional[Callable]) -> Dict:
        """
        通过调用 `fn` 来实现训练时的前向传播过程；
        注意 Trainer 和 Evaluator 会调用该函数来实现网络的前向传播过程，其中传入该函数的参数 `fn` 是函数 `get_model_call_fn` 所返回的
        函数；

        :param batch: 当前的一个 batch 的数据；可以为字典或者其它类型；
        :param fn: 调用该函数进行一次计算。
        :param signature_fn: 由 Trainer 传入的用于网络前向传播一次的签名函数，因为当 batch 是一个 Dict 的时候，我们会自动调用 auto_param_call
        函数，而一些被包裹的模型需要暴露其真正的函数签名，例如 DistributedDataParallel 的调用函数是 forward，但是需要其函数签名为 model.module.forward；
        :return: 返回由 `fn` 返回的结果（应当为一个 dict 或者 dataclass，但是不需要我们去检查）；
        """
        if self._has_fleetwrapped:
            return self.model(batch, fastnlp_fn=fn, fastnlp_signature_fn=signature_fn,
                              wo_auto_param_call=self.wo_auto_param_call)
        else:
            if isinstance(batch, Dict) and not self.wo_auto_param_call:
                return auto_param_call(fn, batch, signature_fn=signature_fn)
            else:
                return fn(batch)

    def get_model_call_fn(self, fn: str) -> Tuple:
        """
        该函数会接受 Trainer 的 train_fn 或者 Evaluator 的 evaluate_fn，返回一个实际用于调用 driver.model_call 时传入的函数参数；
        该函数会在 Trainer 和 Evaluator 在 driver.setup 函数之后调用；

        之所以设置该函数的目的在于希望将具体的 model_call function 从 driver 中抽离出来，然后将其附着在 Trainer 或者 Evaluator 身上；
        这样是因为在新版的设计中，使用 model 的哪种方法来进行 `train step` 或者 `evaluate step` 是通过额外的参数 `train_fn` 和
         `evaluate_fn` 来确定的，而二者又分别是通过 Trainer 和 Evaluator 来控制的；因此不能将确定具体的 `train step fn` 和
         `evaluate step fn` 的逻辑放在每一个 driver 的初始化的时候（因此在 Trainer 初始化第一个 driver 时，Evaluator 还没有初始化，但是
         `evaluate step fn` 的确定却需要 Evaluator 的初始化），因此我们将这一逻辑抽象到这一函数当中；

        这一函数应当通过参数 `fn` 来判断应当返回的实际的调用的函数，具体逻辑如下所示：
            1. 如果 fn == "train_step" or "evaluate_step"，那么对传入的模型进行检测，如果模型没有定义方法 `fn`，则默认调用模型的 `forward`
             函数，然后给出 warning；
            2. 如果 fn 是其他字符串，那么如果模型没有定义方法 `fn` 则直接报错；
        注意不同的 driver 需要做额外的检测处理，例如在 DDPDriver 中，当传入的模型本身就是 DistributedDataParallel 中，我们只能调用模型的
         forward 函数，因此需要额外的 warning；这一点特别需要注意的问题在于 driver 自己在 setup 时也会对模型进行改变（DDPDriver），因此
         可能需要额外标记最初传入 driver 的模型是哪种形式的；

        :param fn: 应当为一个字符串，该函数通过该字符串判断要返回模型的哪种方法；
        :return: 返回一个元组，包含两个函数，用于在调用 driver.model_call 时传入；
        """
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
        r"""
        根据输入的 dataloader 得到一个 支持分布式 （distributed） 与 可复现的 (reproducible) 的 dataloader。

        :param dataloader: 根据 dataloader 设置其对应的分布式版本以及可复现版本
        :param dist: 应当为一个字符串，其值应当为以下之一：[None, "dist", "unrepeatdist"]；为 None 时，表示不需要考虑当前 dataloader
            切换为分布式状态；为 'dist' 时，表示该 dataloader 应该保证每个 gpu 上返回的 batch 的数量是一样多的，允许出现少量 sample ，在
            不同 gpu 上出现重复；为 'unrepeatdist' 时，表示该 dataloader 应该保证所有 gpu 上迭代出来的数据合并起来应该刚好等于原始的
            数据，允许不同 gpu 上 batch 的数量不一致。其中 trainer 中 kwargs 的参数 `use_dist_sampler` 为 True 时，该值为 "dist"；
            否则为 None ，evaluator 中的 kwargs 的参数 `use_dist_sampler` 为 True 时，该值为 "unrepeatdist"，否则为 None；
        注意当 dist 为 ReproducibleSampler, ReproducibleBatchSampler 时，是断点重训加载时 driver.load 函数在调用；
        当 dist 为 str 或者 None 时，是 trainer 在初始化时调用该函数；

        :param reproducible: 如果为 False ，不要做任何考虑；如果为 True ，需要保证返回的 dataloader 可以保存当前的迭代状态，使得
            可以可以加载。
        :return: 应当返回一个被替换 sampler 后的新的 dataloader 对象 (注意此处一定需要返回一个新的 dataloader 对象) ；此外，
            如果传入的 dataloader 中是 ReproducibleSampler 或者 ReproducibleBatchSampler 需要重新初始化一个放入返回的
            dataloader 中。如果 dist 为空，且 reproducible 为 False，可直接返回原对象。
        """
        # 暂时不支持iterableDataset
        assert dataloader.dataset_kind != _DatasetKind.ITER, \
                    "FastNLP does not support `IteratorDataset` now."
        # 如果 dist 为 ReproducibleBatchSampler, ReproducibleSampler 说明是在断点重训时 driver.load 函数调用；
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
                raise RuntimeError("It is not allowed to use checkpoint retraining when you initialize fleet out of our "
                                   "control.")
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

    def is_global_zero(self):
        return self.global_rank == 0

    def get_model_no_sync_context(self):
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

    @staticmethod
    def _check_optimizer_legality(optimizers):
        # paddle 存在设置分布式 optimizers 的函数，返回值为 fleet.meta_optimizers.HybridParallelOptimizer
        DistribuedOptimizer = fleet.meta_optimizers.HybridParallelOptimizer
        for each_optimizer in optimizers:
            if not isinstance(each_optimizer, (Optimizer, DistribuedOptimizer)):
                raise ValueError(f"Each optimizer of parameter `optimizers` should be 'paddle.optimizer.Optimizer' type, "
                                f"not {type(each_optimizer)}.")

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
        # 因为设置了CUDA_VISIBLE_DEVICES，可能会引起错误
        return fastnlp_paddle_broadcast_object(obj, src, device=self.data_device, group=group)

    def all_gather(self, obj, group=None) -> List:
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
        return fastnlp_paddle_all_gather(obj, group=group)
