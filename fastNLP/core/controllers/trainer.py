"""
``Trainer`` 是 fastNLP 用于训练模型的专门的训练器，其支持多种不同的驱动模式 ``Driver``，不仅包括最为经常使用的 DDP，而且还支持 jittor 等国产
的训练框架；新版的 fastNLP 新加入了方便的 callback 函数修饰器，并且支持定制用户自己特定的训练循环过程；通过使用该训练器，用户只需要自己实现
模型部分，而将训练层面的逻辑完全地交给 fastNLP；
"""

from typing import Union, Optional, List, Callable, Dict, BinaryIO
from functools import partial
from collections import defaultdict
import copy
from contextlib import contextmanager
from dataclasses import is_dataclass
import os
from pathlib import Path
import io

__all__ = [
    'Trainer',
]

from .loops import Loop, TrainBatchLoop
from .utils import State, TrainerState
from .utils.utils import check_evaluate_every
from .evaluator import Evaluator
from fastNLP.core.controllers.utils.utils import TrainerEventTrigger, _TruncatedDataLoader
from fastNLP.core.callbacks import Callback, CallbackManager
from fastNLP.core.callbacks.callback import _CallbackWrapper
from fastNLP.core.callbacks.callback_manager import prepare_callbacks
from fastNLP.core.callbacks.callback_event import Event
from fastNLP.core.drivers import Driver
from ..drivers.choose_driver import choose_driver
from fastNLP.core.utils import get_fn_arg_names, match_and_substitute_params, nullcontext
from fastNLP.core.utils.utils import _check_valid_parameters_number
from fastNLP.envs import rank_zero_call
from fastNLP.core.log import logger
from fastNLP.envs import FASTNLP_MODEL_FILENAME, FASTNLP_CHECKPOINT_FILENAME
from fastNLP.core.utils.exceptions import EarlyStopException


class Trainer(TrainerEventTrigger):
    _custom_callbacks: dict = defaultdict(list)

    def __init__(
            self,
            model,
            driver,
            train_dataloader,
            optimizers,
            device: Optional[Union[int, List[int], str]] = "cpu",
            n_epochs: int = 20,
            evaluate_dataloaders=None,
            batch_step_fn: Optional[Callable] = None,
            evaluate_batch_step_fn: Optional[Callable] = None,
            train_fn: Optional[str] = None,
            evaluate_fn: Optional[str] = None,
            callbacks: Union[List[Callback], Callback, None] = None,
            metrics: Optional[dict] = None,
            evaluate_every: Optional[Union[int, Callable]] = -1,
            input_mapping: Optional[Union[Callable, Dict]] = None,
            output_mapping: Optional[Union[Callable, Dict]] = None,
            model_wo_auto_param_call: bool = False,
            accumulation_steps: int = 1,
            fp16: bool = False,
            monitor: Union[str, Callable] = None,
            larger_better: bool = True,
            marker: Optional[str] = None,
            **kwargs
    ):
        r"""
        :param model: 训练所需要的模型，例如 ``torch.nn.Module``；

        .. note::

            当使用 pytorch 时，注意参数 ``model`` 在大多数情况下为 ``nn.Module``。但是您仍能够通过使用一些特定的组合来使用情况，如下所示：

            1. 当希望使用 ``DataParallel`` 时，您应当使用 ``TorchSingleDriver``，意味着您在初始化 ``Trainer`` 时参数 ``device`` 不应当为
            一个 ``List``；

            2. 当您选择自己初始化 ``init_process_group`` 时（这种情况要求您传入的 ``model`` 参数一定为 ``DistributedDataParallel``），
            您应当使用 ``TorchDDPDriver``，意味着您需要通过 ``python -m torch.distributed.launch`` 的方式来启动训练，此时参数 ``device``
            应当设置为 None（此时我们会忽略该参数），具体见下面对于参数 ``device`` 的更详细的解释。

        :param driver: 训练模型所使用的具体的驱动模式，应当为以下选择中的一个：["torch"]，之后我们会加入 jittor、paddle 等
            国产框架的训练模式；其中 "torch" 表示使用 ``TorchSingleDriver`` 或者 ``TorchDDPDriver``，具体使用哪一种取决于参数 ``device``
            的设置；
        :param train_dataloader: 训练数据集，注意其必须是单独的一个数据集，不能是 List 或者 Dict；
        :param optimizers: 训练所需要的优化器；可以是单独的一个优化器实例，也可以是多个优化器组成的 List；
        :param device: 该参数用来指定具体训练时使用的机器；注意当该参数仅当您通过 `torch.distributed.launch/run` 启动时可以为 None，
            此时 fastNLP 不会对模型和数据进行设备之间的移动处理，但是你可以通过参数 `input_mapping` 和 `output_mapping` 来实现设备之间
            数据迁移的工作（通过这两个参数传入两个处理数据的函数）；同时你也可以通过在 kwargs 添加参数 "data_device" 来让我们帮助您将数据
            迁移到指定的机器上（注意这种情况理应只出现在用户在 Trainer 实例化前自己构造 DDP 的场景）；

            device 的可选输入如下所示：

            * *str*: 例如 'cpu', 'cuda', 'cuda:0', 'cuda:1' 等；
            * *torch.device*: 例如 'torch.device("cuda:0")'；
            * *int*: 将使用 ``device_id`` 为该值的 ``gpu`` 进行训练；如果值为 -1，那么默认使用全部的显卡，此时使用的 driver 实例是 `TorchDDPDriver`；
            * *list(int)*: 如果多于 1 个device，应当通过该种方式进行设定；注意此时我们一定会使用 ``TorchDDPDriver``，不管您传入的列表的长度是 1 还是其它值；
            * *None*: 仅当用户自己通过训练框架提供的并行训练启动脚本开启 ddp 进程时为 None；

            .. note::

                如果希望使用 ``TorchDDPDriver``，在初始化 ``Trainer`` 时您应当使用::

                    Trainer(driver="torch", device=[0, 1])

                注意如果这时 ``device=[0]``，我们仍旧会使用 ``TorchDDPDriver``。

                如果希望使用 ``TorchSingleDriver``，则在初始化 ``Trainer`` 时您应当使用::

                    Trainer(driver="torch", device=0)

            .. warning::

                注意参数 ``device`` 仅当您通过 pytorch 或者其它训练框架自身的并行训练启动脚本启动 ddp 训练时才允许为 ``None``！

                例如，当您使用::

                    python -m torch.distributed.launch --nproc_per_node 2 train.py

                来使用 ``TorchDDPDriver`` 时，此时参数 ``device`` 不再有效（不管您是否自己初始化 ``init_process_group``），我们将直接
                通过 ``torch.device(f"cuda:{local_rank}")`` 来获取当前进程所使用的的具体的 gpu 设备。因此此时您需要使用 ``os.environ["CUDA_VISIBLE_DEVICES"]``
                来指定要使用的具体的 gpu 设备。

                另一点需要注意的是，当您没有选择自己初始化 ``init_process_group`` 时，我们仍旧会帮助您把模型和数据迁移到当前进程所使用的
                具体的 gpu 设备上。但是如果您选择自己在 ``Trainer`` 初始化前（意味着在 ``driver`` 的 ``setup`` 前）初始化 ``init_process_group``，
                那么对于模型的迁移应当完全由您自己来完成。此时对于数据的迁移，如果您在 ``Trainer`` 初始化时指定了参数 ``data_device``，那么
                我们会将数据迁移到 ``data_device`` 上；如果其为 None，那么将数据迁移到正确的设备上应当由您自己来完成。

                对于使用 ``TorchDDPDriver`` 的更多细节，请见 :class:`fastNLP.core.drivers.torch_driver.TorchDDPDriver`。

        :param n_epochs: 训练总共的 epoch 的数量，默认为 20；
        :param evaluate_dataloaders: 验证数据集，其可以是单独的一个数据集，也可以是多个数据集；当为多个数据集时，注意其必须是 Dict；默认
            为 None；
        :param batch_step_fn: 定制每次训练时前向运行一个 batch 的数据所执行的函数。该函数应接受两个参数为 ``trainer`` 和 ``batch``，
            不需要要返回值；更详细的使用位置和说明请见 :meth:`fastNLP.core.controllers.TrainBatchLoop.batch_step_fn`；
        :param evaluate_batch_step_fn: 定制每次验证时前向运行一个 batch 的数据所执行的函数。该函数应接受的两个参数为 ``evaluator`` 和 ``batch``，
            不需要有返回值；可以参考 :meth:`fastNLP.core.controllers.EvaluateBatchLoop.batch_step_fn`；
        :param train_fn: 用来控制 ``Trainer`` 在训练的前向传播过程中是调用模型的哪一个函数，例如是 ``train_step`` 还是 ``forward``；
            默认为 ``None``，如果该值是 ``None``，那么我们会默认使用 ``train_step`` 当做前向传播的函数，如果在模型的定义类中没有找到该方法，
            则使用模型默认的前向传播函数，例如对于 pytorch 来说就是 ``forward``。

            .. note::
                在 fastNLP 中，对于训练时使用的前向传播函数的查找逻辑如下所示：

                    1. 如果 ``train_fn`` 为 None，那么在 model 的类 Model 中寻找方法 ``Model.train_step``;如果没有找到，那么默认使用 ``Model.forward``；
                    2. 如果 ``train_fn`` 为一个字符串，例如 'my_step_fn'，那么我们首先会在 model 的类 Model 中寻找方法 ``Model.my_step_fn``，
                    如果没有找到，那么会直接报错；

        :param evaluate_fn: 用来控制 ``Trainer`` 中内置的 ``Evaluator`` 在验证的前向传播过程中是调用模型的哪一个函数，应当为 ``None``
            或者一个字符串；其使用方式和 train_fn 类似；具体可见 :class:`fastNLP.core.controllers.Evaluator`；
        :param callbacks: 训练当中触发的 callback 类，该参数应当为一个列表，其中的每一个元素都应当继承 ``Callback`` 类；具体可见
            :class:`fastNLP.core.callbacks.Callback`；
        :param metrics: 用于传给 ``Trainer`` 内部的 ``Evaluator`` 实例来进行训练过程中的验证。其应当为一个字典，其中 key 表示 monitor，
            例如 {"acc1": AccMetric(), "acc2": AccMetric()}；

            目前我们支持的 ``metric`` 的种类有以下几种：

            1. fastNLP 自己的 ``metric``：详见 :class:`fastNLP.core.metrics.Metric`；
            2. torchmetrics；
            3. allennlp.training.metrics；
            4. paddle.metric；

        :param evaluate_every: 用来控制 ``Trainer`` 内部的 ``Evaluator`` 验证的频率，其可以为负数、正数或者函数：

            1. 为负数时表示每隔几个 ``epoch`` evaluate 一次；
            2. 为正数则表示每隔几个 ``batch`` evaluate 一次；
            3. 为函数时表示用户自己传入的用于控制 evaluate 的频率的函数，该函数的应该接受当前 trainer 对象作为参数，并
            返回一个 bool 值，返回为 True 说明需要进行 evaluate ；将在每个 ``batch`` 结束后调用该函数判断是否需要 evaluate；

            .. note::

                如果参数 ``evaluate_every`` 为函数，其应当类似：

                >>> def my_evaluate_every(trainer) -> bool:
                ...     if (trainer.global_forward_batches+1) % 1000 == 0:
                ...         return True
                ...     else:
                ...         return False

                该函数表示当每经过 1000 个 batch，``Trainer`` 中内置的 ``Evaluator`` 就会验证一次；

                另一个需要注意的事情在于该函数会在每一次 batch 的结尾进行调用，当该函数返回 ``True`` 时，``Evaluator`` 才会进行验证；

        :param input_mapping: 应当为一个字典或者一个函数，表示在当前 step 拿到一个 batch 的训练数据后，应当做怎样的映射处理：

            1. 如果 ``input_mapping`` 是一个字典:

                1. 如果此时 batch 也是一个 ``Dict``，那么我们会把 batch 中同样在 ``input_mapping`` 中的 key 修改为 ``input_mapping`` 的对应 ``key`` 的 ``value``；
                2. 如果此时 batch 是一个 ``dataclass``，那么我们会先将其转换为一个 ``Dict``，然后再进行上述转换；
                3. 如果此时 batch 此时是其它类型，那么我们将会直接报错；
            2. 如果 ``input_mapping`` 是一个函数，那么对于取出的 batch，我们将不会做任何处理，而是直接将其传入该函数里；

            注意该参数会被传进 ``Evaluator`` 中；因此你可以通过该参数来实现将训练数据 batch 移到对应机器上的工作（例如当参数 ``device`` 为 ``None`` 时）；
            如果 ``Trainer`` 和 ``Evaluator`` 需要使用不同的 ``input_mapping``, 请使用 ``train_input_mapping`` 与 ``evaluate_input_mapping`` 分别进行设置。

        :param output_mapping: 应当为一个字典或者函数。作用和 ``input_mapping`` 类似，区别在于其用于转换输出：

            1. 如果 ``output_mapping`` 是一个 ``Dict``，那么我们需要模型的输出必须是 ``Dict`` 或者 ``dataclass`` 类型：

                1. 如果此时模型的输出是一个 ``Dict``，那么我们会把输出中同样在 ``output_mapping`` 中的 key 修改为 ``output_mapping`` 的对应 key 的 value；
                2. 如果此时模型的输出是一个 ``dataclass``，那么我们会先将其转换为一个 Dict，然后再进行上述转换；
            2. 如果 ``output_mapping`` 是一个函数，那么我们将会直接将模型的输出传给该函数；

            如果 ``Trainer`` 和 ``Evaluator`` 需要使用不同的 ``output_mapping``, 请使用 ``train_output_mapping`` 与 ``evaluate_output_mapping`` 分别进行设置；

            .. note::

                ``input_mapping`` 和 ``output_mapping`` 与 fastNLP 的一个特殊的概念 **'参数绑定'** 高度相关，它们的存在也是为了 fastNLP
                中的参数匹配能够正确地运行；

                .. todo::
                    之后链接上 参数匹配 的文档；

            .. warning::

                如果 ``Trainer`` 的参数 ``output_mapping`` 不为 ``None``，请保证其返回的一定是一个字典，并且其中含有关键字 **'loss'**；

        :param model_wo_auto_param_call: 是否关闭在训练时调用我们的 ``auto_param_call`` 函数来自动匹配 batch 和前向函数的参数的行为；

            1. 如果该值为 ``False``，并且当 batch 为字典时，我们会根据**前向函数**所需要的参数从 batch 中提取对应的对象，然后传入到**前向函数**中；
            2. 如果该值为 ``True``，那么我们会将 batch 直接透传给模型；

            .. todo::
                之后链接上 参数匹配 的文档；

            函数 ``auto_param_call`` 详见 :func:`fastNLP.core.utils.auto_param_call`；

        :param accumulation_steps: 梯度累积的步数，表示每隔几个 batch 才让优化器迭代一次，默认为 1；
        :param fp16: 是否开启混合精度训练，默认为 False；
        :param monitor: 对于一些特殊的 ``Callback``，例如 :class:`fastNLP.core.callbacks.CheckpointCallback`，它们需要参数 ``monitor``
            来从 ``Evaluator`` 的验证结果中获取当前评测的值，从而来判断是否执行一些特殊的操作。例如，对于 ``CheckpointCallback`` 而言，如果我们
            想要每隔一个 epoch 让 ``Evaluator`` 进行一次验证，然后保存训练以来的最好的结果；那么我们需要这样设置：

            .. code-block::

                trainer = Trainer(
                    ...,
                    metrics={'acc': accMetric()},
                    callbacks=[CheckpointCallback(
                        ...,
                        monitor='acc',
                        topk=1
                    )]
                )

            这意味着对于 ``CheckpointCallback`` 来说，*'acc'* 就是一个监测的指标，用于在 ``Evaluator`` 验证后取出其需要监测的那个指标的值。

            ``Trainer`` 中的参数 ``monitor`` 的作用在于为没有设置 ``monitor`` 参数但是需要该参数的 *callback* 实例设置该值。关于 ``monitor``
            参数更详细的说明，请见 :class:`fastNLP.core.callbacks.CheckpointCallback`；

            注意该参数仅当 ``Trainer`` 内置的 ``Evaluator`` 不为 None 时且有需要该参数但是没有设置该参数的 *callback* 实例才有效；

        :param larger_better: 对于需要参数 ``monitor`` 的 *callback* 来说，``monitor`` 的值是否是越大越好；类似于 ``monitor``，其作用
            在于为没有设置 ``larger_better`` 参数但是需要该参数的 *callback* 实例设置该值；

            注意该参数仅当 ``Trainer`` 内置的 ``Evaluator`` 不为 None 时且有需要该参数但是没有设置该参数的 *callback* 实例才有效；

        :param marker: 用于标记一个 ``Trainer`` 实例，从而在用户调用 ``Trainer.on`` 函数时，标记该函数属于哪一个具体的 ``Trainer`` 实例；默认为 None；

            .. note::

                marker 的使用场景主要在于如果一个脚本中含有多个 ``Trainer`` 实例，并且含有多个使用 ``Trainer.on`` 修饰的函数时，不同的函数属于
                不同的 ``Trainer`` 实例；

                此时，通过将修饰器 ``Trainer.on`` 的参数 ``marker`` 和 ``Trainer`` 的参数 ``marker`` 置为相同，就可以使得该函数只会在这一
                ``Trainer`` 实例中被调用；例如，

                .. code-block::

                    @Trainer.on(Event.on_train_begin(), marker='trainer1')
                    def fn(trainer):
                        ...

                    trainer = Trainer(
                        ...,
                        marker='trainer1'
                    )

                另一点需要说明的是，如果一个被 ``Trainer.on`` 修饰的函数，其修饰时没有指明 ``marker``，那么会将该函数传给代码位于其之后的
                第一个 ``Trainer`` 实例，即使该 ``Trainer`` 实例的 marker 不为 None；这一点详见 :meth:`~fastNLP.core.controllers.Trainer.on`

        :kwargs:
            * *torch_kwargs* -- 用于在指定 ``driver`` 为 'torch' 时设定具体 driver 实例的一些参数：
                * ddp_kwargs -- 用于在使用 ``TorchDDPDriver`` 时指定 ``DistributedDataParallel`` 初始化时的参数；例如传入
                 {'find_unused_parameters': True} 来解决有参数不参与前向运算导致的报错等；
                * set_grad_to_none -- 是否在训练过程中在每一次 optimizer 更新后将 grad 置为 None；
                * torch_non_blocking -- 表示用于 pytorch 的 tensor 的 to 方法的参数 non_blocking；
            * *data_device* -- 一个具体的 driver 实例中，有 ``model_device`` 和 ``data_device``，前者表示模型所在的设备，后者表示
             当 ``model_device`` 为 None 时应当将数据迁移到哪个设备；

             .. note::

                注意您在绝大部分情况下不会用到该参数！

                1. 当 driver 实例的 ``model_device`` 不为 None 时，该参数无效；
                2. 对于 pytorch，仅当用户自己通过 ``python -m torch.distributed.launch`` 并且自己初始化 ``init_process_group`` 时，
                driver 实例的 ``model_device`` 才会为 None；

            * *use_dist_sampler* -- 表示是否使用分布式的 ``sampler``。在多卡时，分布式 ``sampler`` 将自动决定每张卡上读取的 sample ，使得一个 epoch
             内所有卡的 sample 加起来为一整个数据集的 sample。默认会根据 driver 是否为分布式进行设置。
            * *evaluate_use_dist_sampler* -- 表示在 ``Evaluator`` 中在使用分布式的时候是否将 dataloader 的 ``sampler`` 替换为分布式的 ``sampler``；默认为 ``True``；
            * *output_from_new_proc* -- 应当为一个字符串，表示在多进程的 driver 中其它进程的输出流应当被做如何处理；其值应当为以下之一：
             ["all", "ignore", "only_error"]；当该参数的值不是以上值时，该值应当表示一个文件夹的名字，我们会将其他 rank 的输出流重定向到
             log 文件中，然后将 log 文件保存在通过该参数值设定的文件夹中；默认为 "only_error"；

             注意该参数仅当使用分布式的 ``driver`` 时才有效，例如 ``TorchDDPDriver``；
            * *progress_bar* -- 以哪种方式显示 progress ，目前支持[None, 'raw', 'rich', 'auto'] 或者 RichCallback, RawTextCallback对象，
             默认为 auto , auto 表示如果检测到当前 terminal 为交互型则使用 RichCallback，否则使用 RawTextCallback对象。如果
             需要定制 progress bar 的参数，例如打印频率等，可以传入 RichCallback, RawTextCallback 对象。
            * *train_input_mapping* -- 与 input_mapping 一致，但是只用于 ``Trainer`` 中。与 input_mapping 互斥。
            * *train_output_mapping* -- 与 output_mapping 一致，但是只用于 ``Trainer`` 中。与 output_mapping 互斥。
            * *evaluate_input_mapping* -- 与 input_mapping 一致，但是只用于 ``Evaluator`` 中。与 input_mapping 互斥。
            * *evaluate_output_mapping* -- 与 output_mapping 一致，但是只用于 ``Evaluator`` 中。与 output_mapping 互斥。

        .. note::
            ``Trainer`` 是通过在内部直接初始化一个 ``Evaluator`` 来进行验证；
            ``Trainer`` 内部的 ``Evaluator`` 默认是 None，如果您需要在训练过程中进行验证，你需要保证这几个参数得到正确的传入：

            必须的参数：1. ``metrics``；2. ``evaluate_dataloaders``；

            可选的其它参数：1. ``evaluate_batch_step_fn；2. ``evaluate_fn``；3. ``evaluate_every``；4. ``input_mapping``；
            5. ``output_mapping``； 6. ``model_wo_auto_param_call``；7. ``fp16``；8. ``monitor``；9. ``larger_better``；

        .. warning::

            如果 ``Trainer`` 中内置的 ``Evaluator`` 实例不为 ``None``，那么需要注意 ``Trainer`` 中的一些参数是与 ``Evaluator`` 一致的，它们分别为：

            1. ``Evaluator`` 在初始化时的 ``driver`` 参数是 ``Trainer`` 中已经实例化过的 driver；这一点使得一些参数对于 ``Trainer`` 内部的
            ``Evaluator`` 没有用处，例如 ``device``，``torch_kwargs``，``data_device`` 和 ``output_from_new_proc`` 等；
            2. ``input_mapping``，``output_mapping``，``model_wo_auto_param_call`` 和 ``fp16`` 是 ``Trainer`` 和其内部默认的
            ``Evaluator`` 是一致的；

            当然，对于 ``input_mapping`` 和 ``output_mapping``，您可以通过添加 ``kwargs`` 中的参数 ``evaluate_input_mapping`` 和
            ``evaluate_output_mapping`` 来单独为 ``Evaluator`` 进行更细致的订制。

            另一方面，注意一些专门独属于 ``Evaluator`` 的参数仅当 ``Evaluator`` 不为 None 时才会生效。

        """
        self.model = model
        self.marker = marker
        if isinstance(driver, str):
            self.driver_name = driver
        else:
            self.driver_name = driver.__class__.__name__
        self.device = device
        if train_dataloader is None:
            raise ValueError("Parameter `train_dataloader` can not be None.")
        self.train_dataloader = train_dataloader
        self.evaluate_dataloaders = evaluate_dataloaders
        self.optimizers = optimizers
        self.fp16 = fp16

        train_input_mapping = kwargs.get('train_input_mapping', None)
        train_output_mapping = kwargs.get('train_output_mapping', None)
        evaluate_input_mapping = kwargs.get('evaluate_input_mapping', None)
        evaluate_output_mapping = kwargs.get('evaluate_output_mapping', None)

        train_input_mapping, train_output_mapping, evaluate_input_mapping, evaluate_output_mapping = \
            _get_input_output_mapping(input_mapping, output_mapping, train_input_mapping, train_output_mapping,
                                      evaluate_input_mapping, evaluate_output_mapping)

        self.input_mapping = train_input_mapping
        self.output_mapping = train_output_mapping
        self.evaluate_fn = evaluate_fn

        self.batch_step_fn = batch_step_fn
        if batch_step_fn is not None:
            _check_valid_parameters_number(batch_step_fn, ['trainer', 'batch'], fn_name='batch_step_fn')
            self.check_batch_step_fn = partial(self._check_callback_called_legality, check_mode=True)
        else:
            self.check_batch_step_fn = lambda *args, **kwargs: ...
        # 该变量表示是否检测过 `train_batch_loop`，主要用于当用户通过属性替换的方式使用自己定制的 `train_batch_loop` 时，我们需要检测
        #  用户是否正确地调用了 callback 函数以及是否正确地更新了 `trainer_state` 的状态；
        # 我们将其默认值置为 True，这表示默认的 `train_batch_loop` 已经检测过，不需要再进行检测；
        # 我们只会在第一个 epoch 运行完后进行检测，之后的 epoch 不会再进行检测；
        self.has_checked_train_batch_loop = True
        self._train_batch_loop = TrainBatchLoop(batch_step_fn=batch_step_fn)

        if not isinstance(accumulation_steps, int):
            raise ValueError("Parameter `accumulation_steps` can only be `int` type.")
        elif accumulation_steps < 0:
            raise ValueError("Parameter `accumulation_steps` can only be bigger than 0.")
        self.accumulation_steps = accumulation_steps

        # todo 思路大概是，每个driver提供一下自己的参数是啥（需要对应回初始化的那个），然后trainer/evalutor在初始化的时候，就检测一下自己手上的参数和driver的是不是一致的，不一致的地方需要warn用户说这些值driver不太一样。感觉可以留到后面做吧
        self.driver = choose_driver(
            model=model,
            driver=driver,
            train_dataloader=train_dataloader,
            optimizers=optimizers,
            device=device,
            n_epochs=n_epochs,
            evaluate_dataloaders=evaluate_dataloaders,
            batch_step_fn=batch_step_fn,
            evaluate_batch_step_fn=evaluate_batch_step_fn,
            evaluate_fn=evaluate_fn,
            callbacks=callbacks,
            metrics=metrics,
            evaluate_every=evaluate_every,
            input_mapping=train_input_mapping,
            output_mapping=train_output_mapping,
            model_wo_auto_param_call=model_wo_auto_param_call,
            accumulation_steps=accumulation_steps,
            fp16=fp16,
            marker=marker,
            **kwargs
        )
        self.driver.set_optimizers(optimizers=optimizers)

        # 根据 progress_bar 参数选择 ProgressBarCallback
        callbacks = prepare_callbacks(callbacks, kwargs.get('progress_bar', 'auto'))
        # 初始化 callback manager；
        self.callback_manager = CallbackManager(callbacks)
        # 添加所有的函数式 callbacks；
        self._fetch_matched_fn_callbacks()
        # 添加所有的类 callbacks；
        self.callback_manager.initialize_class_callbacks()

        # 初始化 state，包括提供给用户的接口和我们自己使用的接口；
        self.state = State()
        self.trainer_state = TrainerState(
            n_epochs=n_epochs,
            cur_epoch_idx=0,
            global_forward_batches=0,
            batch_idx_in_epoch=0,
            num_batches_per_epoch=None,  # 会在具体的 train_batch_loop 中进行初始化；
            total_batches=None
        )

        if metrics is None and evaluate_dataloaders is not None:
            raise ValueError("You have set 'evaluate_dataloaders' but forget to set 'metrics'.")

        if metrics is not None and evaluate_dataloaders is None:
            raise ValueError("You have set 'metrics' but forget to set 'evaluate_dataloaders'.")

        self.metrics = metrics
        self.evaluate_every = evaluate_every

        self.driver.setup()
        self.driver.barrier()

        use_dist_sampler = kwargs.get("use_dist_sampler", self.driver.is_distributed())
        if use_dist_sampler:
            _dist_sampler = "dist"
        else:
            _dist_sampler = None

        self.evaluator = None
        self.monitor = monitor
        self.larger_better = larger_better
        if metrics is not None and evaluate_dataloaders is not None:
            check_evaluate_every(evaluate_every)
            progress_bar = kwargs.get('progress_bar', 'auto')  # 如果不为
            if not (isinstance(progress_bar, str) or progress_bar is None): # 应该是ProgressCallback，获取其名称。
                progress_bar = progress_bar.name
            self.evaluator = Evaluator(model=model, dataloaders=evaluate_dataloaders, metrics=metrics,
                                       driver=self.driver, evaluate_batch_step_fn=evaluate_batch_step_fn,
                                       evaluate_fn=evaluate_fn, input_mapping=evaluate_input_mapping,
                                       output_mapping=evaluate_output_mapping, fp16=fp16, verbose=0,
                                       use_dist_sampler=kwargs.get("evaluate_use_dist_sampler", None),
                                       progress_bar=progress_bar)

        if train_fn is not None and not isinstance(train_fn, str):
            raise TypeError("Parameter `train_fn` can only be `str` type when it is not None.")
        self._train_step, self._train_step_signature_fn = self.driver.get_model_call_fn("train_step" if train_fn is None else train_fn)
        self.train_fn = train_fn

        self.dataloader = self.train_dataloader
        self.driver.set_deterministic_dataloader(self.dataloader)

        self.dataloader = self.driver.set_dist_repro_dataloader(dataloader=self.train_dataloader, dist=_dist_sampler,
                                                                reproducible=self.callback_manager._need_reproducible_sampler)

        _torch_kwargs = kwargs.get("torch_kwargs", {})
        self.set_grad_to_none = _torch_kwargs.get("set_grad_to_none", True)

        self.evaluate_batch_step_fn = evaluate_batch_step_fn
        self.kwargs = kwargs

        self.on_after_trainer_initialized(self.driver)
        self.driver.barrier()

    def run(self, num_train_batch_per_epoch: int = -1, num_eval_batch_per_dl: int = -1,
            num_eval_sanity_batch: int = 2, resume_from: str = None, resume_training: bool = True,
            catch_KeyboardInterrupt=None):
        r"""
        注意如果是断点重训的第一次训练，即还没有保存任何用于断点重训的文件，那么其应当置 resume_from 为 None，并且使用 ModelCheckpoint
        去保存断点重训的文件；
        :param num_train_batch_per_epoch: 每个 epoch 运行多少个 batch 即停止，-1 为根据 dataloader 有多少个 batch 决定。
        :param num_eval_batch_per_dl: 每个 evaluate dataloader 运行多少个 batch 停止，-1 为根据 dataloader 有多少个 batch 决定。
        :param num_eval_sanity_batch: 在训练之前运行多少个 evaluation batch 来检测一下 evaluation 是否有错误。为 0 表示不检测。
        :param resume_from: 从哪个路径下恢复 trainer 的状态
        :param resume_training: 是否按照 checkpoint 中训练状态恢复。如果为 False，则只恢复 model 和 optimizers 的状态。
        :param catch_KeyboardInterrupt: 是否捕获KeyboardInterrupt, 如果捕获的话，不会抛出一场，trainer.run()之后的代码会继续运
        行。默认如果非 distributed 的 driver 会 catch ，distributed 不会 catch （无法 catch ）
        :return:
        """

        if catch_KeyboardInterrupt is None:
            catch_KeyboardInterrupt = not self.driver.is_distributed()
        else:
            if self.driver.is_distributed():
                if catch_KeyboardInterrupt:
                    logger.rank_zero_warning("Parameter `catch_KeyboardInterrupt` can only be False when you are using multi-device "
                                   "driver. And we are gonna to set it to False.")
                catch_KeyboardInterrupt = False

        self._set_num_eval_batch_per_dl(num_eval_batch_per_dl)

        if resume_from is not None:
            if os.path.exists(resume_from):
                self.load(resume_from, resume_training=resume_training)
            else:
                raise FileNotFoundError("You are using `resume_from`, but we can not find your specific file.")

        if self.evaluator is not None and num_eval_sanity_batch != 0:
            logger.info(f"Running evaluator sanity check for {num_eval_sanity_batch} batches.")
            self.on_sanity_check_begin()
            sanity_check_res = self.evaluator.run(num_eval_batch_per_dl=num_eval_sanity_batch)
            self.on_sanity_check_end(sanity_check_res)

        if num_train_batch_per_epoch != -1:
            self.dataloader = _TruncatedDataLoader(self.dataloader, num_train_batch_per_epoch)

        self.num_batches_per_epoch = len(self.dataloader)
        self.total_batches = self.num_batches_per_epoch * self.n_epochs
        self.global_forward_batches = self.num_batches_per_epoch * self.cur_epoch_idx + self.batch_idx_in_epoch

        try:
            self.on_train_begin()
            self.driver.barrier()
            self.driver.zero_grad(self.set_grad_to_none)
            while self.cur_epoch_idx < self.n_epochs:
                # 这个是防止在 Trainer.load 之后还没结束当前 epoch 又继续 save
                self.start_batch_idx_in_epoch = self.trainer_state.batch_idx_in_epoch
                self.driver.set_model_mode("train")
                self.on_train_epoch_begin()
                self.driver.set_sampler_epoch(self.dataloader, self.cur_epoch_idx)
                self.train_batch_loop.run(self, self.dataloader)
                if not self.has_checked_train_batch_loop:
                    self._check_train_batch_loop_legality()
                self.cur_epoch_idx += 1
                self.on_train_epoch_end()
                self.driver.barrier()
                self.epoch_evaluate()
                self.driver.barrier()

        except EarlyStopException as e:
            logger.info(f"Catch early stop exception: {e.msg}.")
            self.on_exception(e)
        except KeyboardInterrupt as e:
            self.driver.on_exception()
            self.on_exception(e)
            if not catch_KeyboardInterrupt:
                raise e
        except RuntimeError as e:
            if 'torch' in self.driver_name.lower():  # 如果是 torch ，需要检测一下 find_unused_parameters
                if 'find_unused_parameters' in e.args[0]:
                    logger.error("You may need to pass `torch_ddp_kwargs={'find_unused_parameters': True}` in the "
                                 "Trainer initialization to avoid this error.")
            self.driver.on_exception()
            self.on_exception(e)
            raise e
        except BaseException as e:
            self.driver.on_exception()
            self.on_exception(e)
            raise e
        finally:
            self.on_train_end()

    def _set_num_eval_batch_per_dl(self, num_eval_batch_per_dl):
        def _evaluate_fn(trainer: Trainer, evaluate_fn: Callable) -> None:
            trainer.on_evaluate_begin()
            _evaluate_res: dict = evaluate_fn()
            trainer.on_evaluate_end(_evaluate_res)

        if self.evaluator is not None:
            self.run_evaluate = partial(_evaluate_fn, self, partial(self.evaluator.run, num_eval_batch_per_dl))

    def step_evaluate(self):
        """
        在每个 batch 结束后调用，根据设置执行 evaluate 。

        :return:
        """
        if self.evaluator is not None:
            if callable(self.evaluate_every):
                if self.evaluate_every(self):
                    self.run_evaluate()
            elif self.evaluate_every > 0 and self.global_forward_batches % self.evaluate_every == 0:
                self.run_evaluate()

    def epoch_evaluate(self):
        """
        在每个 epoch 结束后调用，根据设置执行 evaluate 。

        :return:
        """
        if self.evaluator is not None:
            if isinstance(self.evaluate_every, int) and self.evaluate_every < 0:
                evaluate_every = -self.evaluate_every
                if self.cur_epoch_idx % evaluate_every == 0:
                    self.run_evaluate()

    def add_callback_fn(self, event: Event, fn: Callable):
        r"""
        在初始化一个 trainer 实例后，用户可以使用这一函数来方便地添加 callback 函数；
        这一函数应当交给具体的 trainer 实例去做，因此不需要 `mark` 参数；

        :param event: 特定的 callback 时机，用户需要为该 callback 函数指定其属于哪一个 callback 时机；
        :param fn: 具体的 callback 函数；
        """
        if not isinstance(event, Event):
            raise ValueError("parameter event should only be `Event` type.")

        _custom_callback = _CallbackWrapper(event, fn)
        self.callback_manager.dissect_one_callback(_custom_callback)

    @classmethod
    def on(cls, event: Event, marker: Optional[str] = None):
        r"""
        函数修饰器，用户可以使用该函数来方便地将一个函数转变为 callback 函数，从而进行训练流程中的控制；
        支持的 event 时机有以下这些，其执行的时机顺序也如下所示。每个时机装饰的函数应该接受的参数列表也如下所示，例如::

            Trainer.__init__():
                on_after_trainer_initialized(trainer, driver)
            Trainer.run():
                if num_eval_sanity_batch>0:
                    on_sanity_check_begin(trainer)  # 如果设置了num_eval_sanity_batch
                    on_sanity_check_end(trainer, sanity_check_res)
                try:
                    on_train_begin(trainer)
                    while cur_epoch_idx < n_epochs:
                        on_train_epoch_begin(trainer)
                        while batch_idx_in_epoch<=num_batches_per_epoch:
                            on_fetch_data_begin(trainer)
                            batch = next(dataloader)
                            on_fetch_data_end(trainer)
                            on_train_batch_begin(trainer, batch, indices)
                            on_before_backward(trainer, outputs)  # 其中 outputs 是经过 output_mapping（如果设置了） 后的，否则即为 model 的输出。
                            on_after_backward(trainer)
                            on_before_zero_grad(trainer, optimizers)  # 实际调用受到 accumulation_steps 影响
                            on_after_zero_grad(trainer, optimizers)  # 实际调用受到 accumulation_steps 影响
                            on_before_optimizers_step(trainer, optimizers)  # 实际调用受到 accumulation_steps 影响
                            on_after_optimizers_step(trainer, optimizers)  # 实际调用受到 accumulation_steps 影响
                            on_train_batch_end(trainer)
                        on_train_epoch_end(trainer)
                except BaseException:
                    self.on_exception(trainer, exception)
                finally:
                    on_train_end(trainer)

            其它 callback 例如 on_evaluate_begin(trainer)/on_evaluate_end(trainer, results)/on_save_model(trainer)/
            on_load_model(trainer)/on_save_checkpoint(trainer)/on_load_checkpoint(trainer)将根据需要在Trainer.run()中
            特定的时间调用。

        Example::

            from fastNLP import Event
            @Trainer.on(Event.on_save_model())
            def do_something_1(trainer):
                # do something
            # 以上函数会在 Trainer 保存模型时执行。

            @Trainer.on(Event.on_save_model(once=True))
            def do_something_2(trainer):
                # do something
            # 以上函数会在 Trainer 保存模型时执行，但只执行一次。

            @Trainer.on(Event.on_train_batch_begin(every=2))
            def do_something_3(trainer, batch, indices):
                # do something
            # 以上函数会在 Trainer 每个新的 batch 开始的时候执行，但是是两个 batch 才执行一次。

        .. note::


            例如：

            .. code-block::

                @Trainer.on(Event.on_train_begin())
                def fn1(trainer):
                    ...

                @Trainer.on(Event.on_train_epoch_begin())
                def fn2(trainer):
                    ...

                trainer1 = Trainer(
                    ...,
                    marker='trainer1'
                )

                @Trainer.on(Event.on_fetch_data_begin())
                def fn3(trainer):
                    ...

                trainer2 = Trainer(
                    ...,
                    marker='trainer2'
                )


        注意如果你使用该函数修饰器来为你的训练添加 callback，请务必保证你加入 callback 函数的代码在实例化 `Trainer` 之前；

        :param event: 特定的 callback 时机，用户需要为该 callback 函数指定其属于哪一个 callback 时机。每个时机运行的函数应该包含
            特定的参数，可以通过上述说明查阅。
        :param marker: 用来标记该 callback 函数属于哪几个具体的 trainer 实例；两个特殊情况：1.当 `marker` 为 None（默认情况）时，
         表示该 callback 函数只属于代码下方最近的一个 trainer 实例；2.当 `marker` 为 'all' 时，该 callback 函数会被所有的 trainer
         实例使用；
        :return: 返回原函数；
        """

        def wrapper(fn: Callable) -> Callable:
            callback_fn_args = get_fn_arg_names(getattr(Callback, event.value))[1:]
            _check_valid_parameters_number(fn, callback_fn_args)
            cls._custom_callbacks[marker].append((event, fn))
            return fn

        return wrapper

    def _fetch_matched_fn_callbacks(self):
        """
        因为对于使用装饰器加入的函数 callback，我们是加在类属性中，因此在初始化一个具体的 trainer 实例后，我们需要从 Trainer 的
        callback 类属性中将属于其的 callback 函数拿到，然后加入到 callback_manager 中；
        """
        _own_callbacks: List = copy.deepcopy(self._custom_callbacks["all"])
        _own_callbacks.extend(self._custom_callbacks[None])
        logger.debug(f"Get {len(_own_callbacks)} callback fns through Trainer.on().")
        self._custom_callbacks[None] = []
        if self.marker is not None:
            if len(self._custom_callbacks[self.marker]) == 0:
                logger.info(f"You have set `trainer.marker = {self.marker}`, but there are no callback function matched "
                      f"`{self.marker}` that is added through function `Trainer.on`")
            _own_callbacks += self._custom_callbacks[self.marker]
        for each_callback in _own_callbacks:
            self.add_callback_fn(*each_callback)

    def _check_callback_called_legality(self, check_mode: bool = True):
        """
        1. 函数的调用时机：
            当检测 'batch_step_fn' 时，这个函数应当在 'train_batch_loop.run' 的 while 循环的最后进行调用；
            当检测 'TrainBatchLoop' 时，这个函数应当在每一个 epoch 的最后进行调用；

        2. 函数作用
            这一函数的作用在于检查用户定制的 batch_step_fn / TrainBatchLoop 是否能够正确地调用 callback 函数，更准确地说，当用户实际
            定制了 ("on_before_backward", "on_after_backward", "on_before_optimizers_step", "on_after_optimizers_step", "on_before_zero_grad",
            "on_after_zero_grad") /
            ("on_fetch_data_begin", "on_fetch_data_end", "on_train_batch_begin", "on_train_batch_end",
             "on_before_backward", "on_after_backward", "on_before_optimizers_step", "on_after_optimizers_step", "on_before_zero_grad",
             "on_after_zero_grad")
            这些 callabck_fn 后，如果其同样也定制了 batch_step_fn / TrainBatchLoop，那么其有可能忘记了在自己的 batch_step_fn 中
            上述的这些 callback 函数，而这个函数的作用就在于检测用户是否产生了这一行为；

        注意，这一函数只会在 batch_step_fn 不为 None 时或者 TrainBatchLoop 没有被替换时才会被调用；

        :param check_mode: 用来判断该函数是用来检测 'batch_step_fn' 还是用来检测 'TrainBatchLoop' 的参数，为 True 时表示检测
         'batch_step_fn'，为 False 时表示检测 'TrainBatchLoop'；
        """
        if check_mode:
            callbacks = ("on_before_backward", "on_after_backward", "on_before_optimizers_step", "on_after_optimizers_step",
                         "on_before_zero_grad", "on_after_zero_grad")
        else:
            callbacks = ("on_fetch_data_begin", "on_fetch_data_end", "on_train_batch_begin", "on_train_batch_end",
                         "on_before_backward", "on_after_backward", "on_before_optimizers_step", "on_after_optimizers_step",
                         "on_before_zero_grad", "on_after_zero_grad")
        _not_called_callback_fns = []
        for each_callback_fn in callbacks:
            if each_callback_fn in self.callback_manager.callback_fns:
                if self.callback_manager.callback_counter[each_callback_fn] == 0:
                    _not_called_callback_fns.append(each_callback_fn)

        if check_mode:
            logger.rank_zero_warning("You have customized your 'batch_step_fn' in the 'train_batch_loop' and also use these "
                           f"callback_fns: {_not_called_callback_fns}, but it seems that"
                           "you don't call the corresponding callback hook explicitly in your 'batch_step_fn'.")
            # 对于 'batch_step_fn' 来讲，其只需要在第一次的 step 后进行检测即可，因此在第一次检测后将 check_batch_step_fn 置为 pass
            #  函数；
            self.check_batch_step_fn = lambda *args, **kwargs: ...
        else:
            logger.warning("You have customized your 'TrainBatchLoop' and also use these callback_fns: "
                           f"{_not_called_callback_fns}, but it seems that"
                           "you don't call the corresponding callback hook explicitly in your 'batch_step_fn'.")

    def _check_train_batch_loop_legality(self):
        r"""
        该函数用于检测用户定制的 `train_batch_loop` 是否正确地调用了 callback 函数以及是否正确地更新了 `trainer_state` 的状态；
        该函数仅当用户通过属性更换用自己的定制的 `train_batch_loop` 替换了默认的 `TrainBatchLoop` 对象后才会被调用；
        当被调用时，该函数仅当第一次被调用时被调用；
        """
        # 1. 检测用户定制的 `train_batch_loop` 是否正确地调用了 callback 函数；
        self._check_callback_called_legality(check_mode=False)

        # 2. 检测用户定制的 `train_batch_loop` 是否正确地更新了 `trainer_state` 的状态；
        #  因为该检测函数只会在第一个 epoch 运行完后调用，因此我们只需要检测这些 `trainer_state` 的值是否正确即可；
        if self.batch_idx_in_epoch == 0:
            logger.warning("You have customized your `train_batch_loop`, but it seemed that you forget to update the "
                           "`trainer_state.batch_idx_in_epoch` in your process of training. Look the origin class "
                           "`TrainBatchLoop`.")
        if self.global_forward_batches == 0:
            logger.warning("You have customized your `train_batch_loop`, but it seemed that you forget to update the "
                           "`trainer_state.global_forward_batches` in your process of training. Look the origin class "
                           "`TrainBatchLoop`.")
        self.has_checked_train_batch_loop = True

    """ Trainer 需要的一些 property """
    @property
    def driver(self):
        return self._driver

    @driver.setter
    def driver(self, driver: Driver):
        self._driver = driver

    @property
    def train_batch_loop(self):
        return self._train_batch_loop

    @train_batch_loop.setter
    def train_batch_loop(self, loop: Loop):
        self.has_checked_train_batch_loop = False
        if self.batch_step_fn is not None:
            logger.warning("`batch_step_fn` was customized in the Trainer initialization, it will be ignored "
                           "when the `train_batch_loop` is also customized.")
            # 如果用户定制了 TrainBatchLoop，那么我们不需要再专门去检测 batch_step_fn，因为该函数一定会被忽略；
            self.check_batch_step_fn = lambda *args, **kwargs: ...
        self._train_batch_loop = loop

    def save_model(self, folder: Union[str, os.PathLike, BinaryIO, io.BytesIO], only_state_dict: bool = False,
                   model_save_fn: Optional[Callable] = None, **kwargs):
        r"""
        用于帮助用户保存模型的辅助函数，具体实际的保存模型的操作由具体的 driver 实现；

        :param folder: 保存模型的文件夹。如果没有传入 model_save_fn 参数，则在这个文件夹下创建 fastnlp_model.pkl.tar 文件。
        :param only_state_dict: 仅在 model_save_fn 为空时，有效。是否只保存模型的 `state_dict`；
        :param model_save_fn: 用户自己定制的用来替换该保存函数本身保存逻辑的函数；
        :param kwargs:
        """

        self.on_save_model()
        self.driver.barrier()

        if not isinstance(folder, (io.BytesIO, BinaryIO)):
            if model_save_fn is not None:
                if not callable(model_save_fn):
                    raise ValueError("Parameter `model_save_fn` should be `Callable` type when it is not None.")
                rank_zero_call(model_save_fn)(folder)
            else:
                if isinstance(folder, str):
                    folder = Path(folder)
                self.driver.save_model(folder.joinpath(FASTNLP_MODEL_FILENAME), only_state_dict, **kwargs)
        else:
            if model_save_fn is not None:
                raise RuntimeError("It is not allowed to specify a `model_save_fn` parameter with `folder` being "
                                   "`io.BytesIO` type.")
            self.driver.save_model(folder, only_state_dict, **kwargs)
        self.driver.barrier()

    def load_model(self, folder: Union[str, Path, BinaryIO, io.BytesIO], only_state_dict: bool = True,
                   model_load_fn: Optional[Callable] = None, **kwargs):
        """
        加载模型

        :param folder: 读取 model 的文件夹，默认会尝试读取该文件夹下的 fastnlp_model.pkl.tar 文件。在 model_load_fn 不为空时，
            直接将该 folder 传递到 model_load_fn 中。
        :param only_state_dict: 要读取的文件中是否仅包含模型权重。在 model_load_fn 不为 None 时，该参数无意义。
        :param model_load_fn: callable 的函数，接受一个 folder 作为参数，不返回任何内容。
        :param kwargs:
        :return:
        """
        self.on_load_model()
        self.driver.barrier()
        if not isinstance(folder, (io.BytesIO, BinaryIO)):
            try:
                if model_load_fn is not None:
                    if not callable(model_load_fn):
                        raise ValueError("Parameter `model_save_fn` should be `Callable` type when it is not None.")
                    model_load_fn(folder)
                else:
                    if isinstance(folder, str):
                        folder = Path(folder)
                    self.driver.load_model(folder.joinpath(FASTNLP_MODEL_FILENAME), only_state_dict, **kwargs)
            except FileNotFoundError as e:
                if FASTNLP_MODEL_FILENAME not in os.listdir(folder):
                    logger.error(f"fastNLP model checkpoint file:{FASTNLP_MODEL_FILENAME} is not found in {folder}.")
                raise e
        else:
            if model_load_fn is not None:
                raise RuntimeError("It is not allowed to specify a `model_save_fn` parameter with `folder` being "
                                   "`io.BytesIO` type.")
            self.driver.load_model(folder, only_state_dict, **kwargs)
        self.driver.barrier()

    def save(self, folder: Union[str, Path], only_state_dict: bool = True, model_save_fn: Optional[Callable] = None, **kwargs):
        r"""
        用于断点重训 Trainer 的保存函数。

        :param folder: 保存在哪个文件夹下，会在该文件下声称两个文件：fastnlp_checkpoint.pkl.tar 与 fastnlp_model.pkl.tar 。
            如果 model_save_fn 不为空，则没有 fastnlp_model.pkl.tar 文件。
        :param only_state_dict: 当 model_save_fn 为空时有效，表明是否仅保存模型的权重。
        :param model_save_fn: 如果模型保存比较特殊，可以传入该函数自定义保存过程，输入应该接受一个文件夹（实际上就是接受上面的 folder
            参数），不必返回任何东西。
        :param kwargs:
        :return:
        """
        self.driver.barrier()

        # 1. callback states 和 每一个callback的具体 callback 函数的 filter 的状态；
        # 2. trainer_state；
        states = {"callback_states": self.on_save_checkpoint(),
                  "trainer_state": self.trainer_state.state_dict(),
                  'num_consumed_batches': self.batch_idx_in_epoch - getattr(self, 'start_batch_idx_in_epoch', 0)
                  }

        if isinstance(folder, str):
            folder = Path(folder)

        if model_save_fn is not None:
            if not callable(model_save_fn):
                raise ValueError("Parameter `model_save_fn` should be `Callable` type when it is not None.")
            rank_zero_call(model_save_fn)(folder)
            self.driver.save(folder=folder, dataloader=self.dataloader, states=states, should_save_model=False, **kwargs)
        else:
            self.driver.save(folder=folder, dataloader=self.dataloader, states=states,
                             only_state_dict=only_state_dict, should_save_model=True, **kwargs)

        self.driver.barrier()

    def load(self, folder: str, resume_training: bool = True, only_state_dict: bool = True,
             model_load_fn: Optional[Callable] = None, **kwargs):
        r"""
        用于断点重训的加载函数；
        注意在 fastNLP 中断点重训的保存和加载逻辑是分开的，因此可能存在一种情况：用户只希望加载一个断点重训的状态，而在之后不再进行断点重训的
        保存；在这种情况下，dataloader 的 sampler 就不一定会被替换成我们的 ReproducibleSampler；

        注意我们目前不支持单卡到多卡的断点重训；

        :param folder: 保存断点重训 states 的文件地址；
        :param resume_training: 是否从上次的 batch 开始训练，或者只从最近的 epoch 开始训练；注意如果 resume_training=True，那么我们
         只会加载 model 和 optimizers 的状态；而其余的对象的值则根据用户的 Trainer 的初始化直接重置；
        :param only_state_dict: 保存的 model 是否只包含了权重。
        :param model_load_fn: 使用的模型加载函数，参数应为一个 文件夹，不返回任何内容。
        """
        self.driver.barrier()
        if isinstance(folder, str):
            folder = Path(folder)

        dataloader = self.dataloader
        if not resume_training:
            dataloader = None
        try:
            if model_load_fn is not None:
                if not callable(model_load_fn):
                    raise ValueError("Parameter `model_save_fn` should be `Callable`.")
                model_load_fn(folder)
                states = self.driver.load(folder=folder, dataloader=dataloader, should_load_model=False, **kwargs)
            else:
                states = self.driver.load(folder=folder, dataloader=dataloader, only_state_dict=only_state_dict, should_load_model=True, **kwargs)
        except FileNotFoundError as e:
            if FASTNLP_CHECKPOINT_FILENAME not in os.listdir(folder) and FASTNLP_MODEL_FILENAME in os.listdir(folder):
                logger.error("It seems that you are trying to load the trainer checkpoint from a model checkpoint folder.")
            elif FASTNLP_CHECKPOINT_FILENAME not in os.listdir(folder):
                logger.error(f"fastNLP Trainer checkpoint file:{FASTNLP_CHECKPOINT_FILENAME} is not found in {folder}.")
            raise e

        if not resume_training:
            return

        self.dataloader = states.pop('dataloader')

        # 1. 恢复 trainer_state 的状态；
        self.trainer_state.load_state_dict(states["trainer_state"])

        # 2. 修改 trainer_state.batch_idx_in_epoch
        # sampler 是类似 RandomSampler 的sampler，不是 batch_sampler；
        # 这里的原则就是应当使得    '还会产生的batch数量' + 'batch_idx_in_epoch' = '原来不断点训练的batch的总数'。其中由于
        #    '还会产生的batch数量' 是由还剩多少 sample 决定的，因此只能通过调整 'batch_idx_in_epoch' 使得等式成立
        self.trainer_state.batch_idx_in_epoch = states.pop('batch_idx_in_epoch')
        # 这个是防止用户在 Trainer.load 之后还没结束当前 epoch 又继续 save
        self.start_batch_idx_in_epoch = self.trainer_state.batch_idx_in_epoch

        # 5. 恢复所有 callback 的状态；
        self.on_load_checkpoint(states["callback_states"])

        self.driver.barrier()

    """ 这四个函数是用来方便用户定制自己的 batch_step_fn（用于替换 train_batch_loop 当中的 batch_step_fn 函数） 的 """

    def train_step(self, batch):
        with self.driver.auto_cast():
            outputs = self.driver.model_call(batch, self._train_step, self._train_step_signature_fn)
            outputs = match_and_substitute_params(self.output_mapping, outputs)
            return outputs

    def backward(self, outputs):
        self.on_before_backward(outputs)
        loss = self.extract_loss_from_outputs(outputs)
        loss = loss / self.accumulation_steps
        self.driver.backward(loss)
        self.on_after_backward()

    def zero_grad(self):
        if (self.global_forward_batches + 1) % self.accumulation_steps == 0:
            self.on_before_zero_grad(self.optimizers)
            self.driver.zero_grad(self.set_grad_to_none)
            self.on_after_zero_grad(self.optimizers)

    def step(self):
        if (self.global_forward_batches + 1) % self.accumulation_steps == 0:
            self.on_before_optimizers_step(self.optimizers)
            self.driver.step()
            self.on_after_optimizers_step(self.optimizers)

    def move_data_to_device(self, batch):
        return self.driver.move_data_to_device(batch)

    @staticmethod
    def extract_loss_from_outputs(outputs):
        r"""
        用来从用户模型的输出对象中抽取 `loss` 对象；
        目前支持 `outputs` 对象为 'Dict' 或者 'dataclass'；

        :return: 返回被抽取出来的 `loss` 对象，如果当前运行的是 'pytorch' 的 `Driver`，那么返回的就是一个 tensor；
        """
        if isinstance(outputs, Dict):
            try:
                loss = outputs["loss"]
            except:
                raise KeyError(f"We cannot find `loss` from your model output(with keys:{outputs.keys()}). Please either "
                               f"directly return it from your model or use `output_mapping` to prepare it.")
        elif is_dataclass(outputs):
            try:
                loss = outputs.loss
            except:
                raise AttributeError("We cannot find `loss` from your model output. Please either directly return it from"
                                     " your model or use `output_mapping` to prepare it.")
        else:
            raise ValueError("The `outputs` from your model could only be of `dataclass` or `Dict` type. Or you can use "
                             "the parameter `output_mapping` to prepare loss.")

        return loss

    @contextmanager
    def get_no_sync_context(self):
        r"""
        用于在梯度累积并且使用 DDP 时，由于在前 `accumulation_steps` - 1 的时间内不需要进行梯度的同步，因此通过使用该 context 上下文
         环境来避免梯度的同步；

        :return: 一个 no_sync 的 context；
        """

        if (self.global_forward_batches + 1) % self.accumulation_steps != 0:
            _no_sync_context = self.driver.get_model_no_sync_context()
        else:
            _no_sync_context = nullcontext

        with _no_sync_context():
            yield

    """ trainer state property """

    @property
    def n_epochs(self) -> int:
        return self.trainer_state.n_epochs

    @n_epochs.setter
    def n_epochs(self, n_epochs: int):
        self.trainer_state.n_epochs = n_epochs

    @property
    def cur_epoch_idx(self) -> int:
        return self.trainer_state.cur_epoch_idx

    @cur_epoch_idx.setter
    def cur_epoch_idx(self, cur_epoch_idx: int):
        self.trainer_state.cur_epoch_idx = cur_epoch_idx

    @property
    def global_forward_batches(self) -> int:
        return self.trainer_state.global_forward_batches

    @global_forward_batches.setter
    def global_forward_batches(self, global_forward_batches: int):
        self.trainer_state.global_forward_batches = global_forward_batches

    @property
    def batch_idx_in_epoch(self) -> int:
        return self.trainer_state.batch_idx_in_epoch

    @batch_idx_in_epoch.setter
    def batch_idx_in_epoch(self, batch_idx_in_epoch: int):
        self.trainer_state.batch_idx_in_epoch = batch_idx_in_epoch

    @property
    def num_batches_per_epoch(self) -> int:
        return self.trainer_state.num_batches_per_epoch

    @num_batches_per_epoch.setter
    def num_batches_per_epoch(self, num_batches_per_epoch: int):
        self.trainer_state.num_batches_per_epoch = num_batches_per_epoch

    @property
    def total_batches(self) -> int:
        return self.trainer_state.total_batches

    @total_batches.setter
    def total_batches(self, total_batches: int):
        self.trainer_state.total_batches = total_batches

    """ driver property """

    @property
    def model_device(self):
        return self.driver.model_device

    @property
    def data_device(self):
        return self.driver.data_device

    """ dataloader property """

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, train_dataloader):
        self._train_dataloader = train_dataloader

    @property
    def evaluate_dataloaders(self):
        return self._evaluate_dataloaders

    @evaluate_dataloaders.setter
    def evaluate_dataloaders(self, evaluate_dataloaders):
        self._evaluate_dataloaders = evaluate_dataloaders


def _get_input_output_mapping(input_mapping, output_mapping, train_input_mapping, train_output_mapping,
                              evaluate_input_mapping, evaluate_output_mapping):
    if train_input_mapping is not None and input_mapping is not None:
        raise ValueError("Parameter `input_mapping` and `train_input_mapping` cannot be set simultaneously.")

    if evaluate_input_mapping is not None and input_mapping is not None:
        raise ValueError("Parameter `input_mapping` and `evaluate_input_mapping` cannot be set simultaneously.")

    if train_output_mapping is not None and output_mapping is not None:
        raise ValueError("Parameter `output_mapping` and `train_output_mapping` cannot be set simultaneously.")

    if evaluate_output_mapping is not None and output_mapping is not None:
        raise ValueError("Parameter `output_mapping` and `evaluate_output_mapping` cannot be set simultaneously.")

    if train_input_mapping is None:
        train_input_mapping = input_mapping
    if evaluate_input_mapping is None:
        evaluate_input_mapping = input_mapping

    if train_output_mapping is None:
        train_output_mapping = output_mapping
    if evaluate_output_mapping is None:
        evaluate_output_mapping = output_mapping

    return train_input_mapping, train_output_mapping, evaluate_input_mapping, evaluate_output_mapping






