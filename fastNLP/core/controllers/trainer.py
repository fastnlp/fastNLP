from typing import Union, Optional, List, Callable, Dict, Sequence, BinaryIO, IO
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
from fastNLP.core.callbacks import Callback, CallbackManager, Events, EventsList
from fastNLP.core.callbacks.callback import _CallbackWrapper
from fastNLP.core.callbacks.callback_events import _SingleEventState
from fastNLP.core.callbacks.progress_callback import choose_progress_callback
from fastNLP.core.drivers import Driver
from fastNLP.core.drivers.utils import choose_driver
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
        `Trainer` 是 fastNLP 用于训练模型的专门的训练器，其支持多种不同的驱动模式，不仅包括最为经常使用的 DDP，而且还支持 jittor 等国产
         的训练框架；新版的 fastNLP 新加入了方便的 callback 函数修饰器，并且支持定制用户自己特定的训练循环过程；通过使用该训练器，用户只需
         要自己实现模型部分，而将训练层面的逻辑完全地交给 fastNLP；

        :param model: 训练所需要的模型，目前支持 pytorch；
        :param driver: 训练模型所使用的具体的驱动模式，应当为以下选择中的一个：["torch", "torch_ddp", ]，之后我们会加入 jittor、paddle
         等国产框架的训练模式；其中 "torch" 表示使用 cpu 或者单张 gpu 进行训练
        :param train_dataloader: 训练数据集，注意其必须是单独的一个数据集，不能是 List 或者 Dict；
        :param optimizers: 训练所需要的优化器；可以是单独的一个优化器实例，也可以是多个优化器组成的 List；
        :param device: 该参数用来指定具体训练时使用的机器；注意当该参数为 None 时，fastNLP 不会将模型和数据进行设备之间的移动处理，但是你
         可以通过参数 `input_mapping` 和 `output_mapping` 来实现设备之间数据迁移的工作（通过这两个参数传入两个处理数据的函数）；同时你也
         可以通过在 kwargs 添加参数 "data_device" 来让我们帮助您将数据迁移到指定的机器上（注意这种情况理应只出现在用户在 Trainer 实例化前
         自己构造 DDP 的多进程场景）；
        device 的可选输入如下所示：
            1. 可选输入：str: ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...] 依次为'cpu'中, 可见的第一个GPU中, 可见的第一个GPU中, 可见的第二个GPU中；
            2. torch.device：将模型装载到torch.device上；
            3. int： 将使用device_id为该值的gpu进行训练；如果值为 -1，那么默认使用全部的显卡，此时是 `TorchDDPDriver`；
            4. list(int)：如果多于1个device，应当通过该种方式进行设定；当 `device` 为一个 list 时，我们默认使用 `TorchDDPDriver`；
            5. None： 为None则不对模型进行任何处理；
        :param n_epochs: 训练总共的 epoch 的数量，默认为 20；
        :param evaluate_dataloaders: 验证数据集，其可以是单独的一个数据集，也可以是多个数据集；当为多个数据集时，注意其必须是 Dict；默认
         为 None；
        :param batch_step_fn: 用来替换 `TrainBatchLoop` 中的 `batch_step_fn` 函数，注意该函数的两个参数必须为 `trainer` 和
         `batch`；默认为 None；
        :param evaluate_batch_step_fn: 用来替换 'Evaluator' 中的 `EvaluateBatchLoop` 中的 `batch_step_fn` 函数，注意该函数的
         两个参数必须为 `evaluator` 和 `batch`；默认为 None；
        :param train_fn: 用来控制 `Trainer` 在训练的前向传播过程中是调用模型的哪一个函数，例如是 `train_step` 还是 `forward`；
         默认为 None，如果该值是 None，那么我们会默认使用 `train_step` 当做前向传播的函数，如果在模型中没有找到该方法，
         则使用模型默认的前向传播函数。
        :param evaluate_fn: 用来控制 `Trainer` 中内置的 `Evaluator` 的模式，应当为 None 或者一个字符串；其使用方式和 train_fn 类似；
         注意该参数我们会直接传给 Trainer 中内置的 Evaluator（如果不为 None）；如果该值为 None ，将首先尝试寻找模型中是否有
         evaluate_step 这个函数，如果没有则使用 forward 函数。
        :param callbacks: 训练当中触发的 callback 类，该参数应当为一个列表，其中的每一个元素都应当继承 `Callback` 类；
        :param metrics: 应当为一个字典，其中 key 表示 monitor，例如 {"acc1": AccMetric(), "acc2": AccMetric()}；
        :param evaluate_every: 可以为负数、正数或者函数；为负数时表示每隔几个 epoch evaluate 一次；为正数则表示每隔几个 batch evaluate 一次；
         为函数时表示用户自己传入的用于控制 Trainer 中的 evaluate 的频率的函数，该函数的应该接受当前 trainer 对象作为参数，并
         返回一个 bool 值，返回为 True 说明需要进行 evaluate ；将在每个 batch 结束后调用该函数判断是否需要 evaluate 。
        :param input_mapping: 应当为一个字典或者一个函数，表示在当前 step 拿到一个 batch 的训练数据后，应当做怎样的映射处理；如果其是
         一个字典，并且 batch 也是一个 `Dict`，那么我们会把 batch 中同样在 input_mapping 中的 key 修改为 input_mapping 的对应 key 的
         value；如果 batch 是一个 `dataclass`，那么我们会先将该 dataclass 转换为一个 Dict，然后再进行上述转换；如果 batch 此时是其它
         类型，那么我们将会直接报错；如果 input_mapping 是一个函数，那么对于取出的 batch，我们将不会做任何处理，而是直接将其传入该函数里；
         注意该参数会被传进 `Evaluator` 中；因此你可以通过该参数来实现将训练数据 batch 移到对应机器上的工作（例如当参数 `device` 为 None 时）；
        :param output_mapping: 应当为一个字典或者函数。作用和 input_mapping 类似，区别在于其用于转换输出；如果 output_mapping 是一个
         函数，那么我们将会直接将模型的输出传给该函数；如果其是一个 `Dict`，那么我们需要 batch 必须是 `Dict` 或者 `dataclass` 类型，
         如果 batch 是一个 `Dict`，那么我们会把 batch 中同样在 output_mapping 中的 key 修改为 output_mapping 的对应 key 的 value；
         如果 batch 是一个 `dataclass`，那么我们会先将该 dataclass 转换为一个 Dict，然后再进行上述转换；
        :param model_wo_auto_param_call: 是否关闭在训练时调用我们的 auto_param_call 来自动匹配 batch 和 forward 函数的参数的行为；
         如果该值为 False，并且当 batch 为字典时，我们会根据 forward 所需要的参数从 batch 中提取对应的对象，传入到 forward 函数中；如果该值
         为 True，那么我们会将 batch 直接透传给模型。注意该参数应用于 `train_step`, `evaluate_step` 和 `test_step`；
        :param accumulation_steps: 梯度累积的步数，表示每隔几个 batch 优化器迭代一次；默认为 1；
        :param fp16: 是否开启混合精度训练；默认为 False；
        :param monitor: 当存在 evaluate_dataloaders 时，默认的 monitor metric 的名字。传入的 callback 如果有 monitor 参数且没有
            在 callback 初始化设定的，将采取这个值。如果在 evaluation 结果中没有找到完全一致的名称，将使用 最短公共字符串算法 找到最匹配
            的那个作为 monitor 。也可以传入一个函数，接受参数为 evaluation 的结果(字典类型)，返回一个 float 值作为 monitor 的结果。
        :param larger_better: monitor 的值是否是越大越好。
        :param marker: 用于标记一个 Trainer 实例，从而在用户调用 `Trainer.on` 函数时，标记该 callback 函数属于哪一个具体的 'trainer' 实例；默认为 None；
        :param kwargs: 一些其它的可能需要的参数；
            torch_non_blocking: 表示用于 pytorch 的 tensor 的 to 方法的参数 non_blocking；
            data_device: 表示如果用户的模型 device （在 Driver 中对应为参数 model_device）为 None 时，我们会将数据迁移到 data_device 上；
             注意如果 model_device 为 None，那么 data_device 不会起作用；
            torch_ddp_kwargs: 用于配置 pytorch 的 DistributedDataParallel 初始化时的参数；仅用于 pytorch ddp 训练。例如传入
                {'find_unused_parameters': True} 来解决有有参数不参与前向运算导致的报错等。
            set_grad_to_none: 是否在训练过程中在每一次 optimizer 更新后将 grad 置为 None；
            use_dist_sampler: 表示是否使用分布式的 sampler 。在多卡时，分布式 sampler 将自动决定每张卡上读取的 sample ，使得一个epoch
                内所有卡的 sample 加起来为一整个数据集的 sample。默认会根据 driver 是否为分布式进行设置。
            evaluate_use_dist_sampler: 表示在 Evaluator 中在使用 分布式 的时候是否将 dataloader 的 sampler 替换为分布式的 sampler；默认为 True；
            output_from_new_proc: 应当为一个字符串，表示在多进程的 driver 中其它进程的输出流应当被做如何处理；其值应当为以下之一：
             ["all", "ignore", "only_error"]；当该参数的值不是以上值时，该值应当表示一个文件夹的名字，我们会将其他 rank 的输出流重定向到
             log 文件中，然后将 log 文件保存在通过该参数值设定的文件夹中；默认为 "only_error"；
            progress_bar: 以哪种方式显示 progress ，目前支持[None, 'raw', 'rich', 'auto'] 或者 RichCallback, RawTextCallback对象，
                默认为 auto , auto 表示如果检测到当前 terminal 为交互型 则使用 RichCallback，否则使用 RawTextCallback对象。如果
                需要定制 progress bar 的参数，例如打印频率等，可以传入 RichCallback, RawTextCallback 对象。
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
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
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
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            model_wo_auto_param_call=model_wo_auto_param_call,
            accumulation_steps=accumulation_steps,
            fp16=fp16,
            marker=marker,
            **kwargs
        )
        self.driver.set_optimizers(optimizers=optimizers)

        # 根据 progress_bar 参数选择 ProgressBarCallback
        progress_bar_callback = choose_progress_callback(kwargs.get('progress_bar', 'auto'))
        if progress_bar_callback is not None:
            if callbacks is None:
                callbacks = []
            elif not isinstance(callbacks, Sequence):
                callbacks = [callbacks]

            callbacks = list(callbacks) + [progress_bar_callback]
        else:
            rank_zero_call(logger.warning)("No progress bar is provided, there will have no information output "
                                           "during training.")
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

        """ 设置内部的 Evaluator """
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
            self.evaluator = Evaluator(
                model=model,
                dataloaders=evaluate_dataloaders,
                metrics=metrics,
                driver=self.driver,
                device=device,
                batch_step_fn=evaluate_batch_step_fn,
                evaluate_fn=evaluate_fn,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                fp16=fp16,
                verbose=0,
                use_dist_sampler=kwargs.get("evaluate_use_dist_sampler", None),
                progress_bar=progress_bar
            )

        if train_fn is not None and not isinstance(train_fn, str):
            raise TypeError("Parameter `train_fn` can only be `str` type when it is not None.")
        self._train_step, self._train_step_signature_fn = self.driver.get_model_call_fn("train_step" if train_fn is None else train_fn)
        self.train_fn = train_fn

        self.dataloader = self.train_dataloader
        self.driver.set_deterministic_dataloader(self.dataloader)

        self.dataloader = self.driver.set_dist_repro_dataloader(dataloader=self.train_dataloader, dist=_dist_sampler,
                                                                reproducible=self.callback_manager._need_reproducible_sampler)

        self.set_grad_to_none = kwargs.get("set_grad_to_none", True)

        self.evaluate_batch_step_fn = evaluate_batch_step_fn
        self.kwargs = kwargs

        self.on_after_trainer_initialized(self.driver)
        self.driver.barrier()

    def run(self, num_train_batch_per_epoch: int = -1, num_eval_batch_per_dl: int = -1,
            num_eval_sanity_batch: int = 2, resume_from: str = None, resume_training: bool = True,
            catch_KeyboardInterrupt=None):
        """
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
                    logger.warning("Parameter `catch_KeyboardInterrupt` can only be False when you are using multi-device "
                                   "driver. And we are gonna to set it to False.")
                catch_KeyboardInterrupt = False

        self._set_num_eval_batch_per_dl(num_eval_batch_per_dl)

        if resume_from is not None:
            if os.path.exists(resume_from):
                self.load(resume_from, resume_training=resume_training)
            else:
                raise FileNotFoundError("You are using `resume_from`, but we can not find your specific file.")

        if self.evaluator is not None and num_eval_sanity_batch > 0:
            logger.info(f"Running evaluator sanity check for {num_eval_sanity_batch} batches.")
            self.on_sanity_check_begin()
            sanity_check_res = self.evaluator.run(num_eval_batch_per_dl=num_eval_sanity_batch)
            self.on_sanity_check_end(sanity_check_res)

        if num_train_batch_per_epoch != -1:
            self.dataloader = _TruncatedDataLoader(self.dataloader, num_train_batch_per_epoch)

        self.num_batches_per_epoch = len(self.dataloader)
        self.total_batches = self.num_batches_per_epoch * self.n_epochs
        self.global_forward_batches = self.num_batches_per_epoch * self.cur_epoch_idx + self.batch_idx_in_epoch
        self.on_train_begin()
        self.driver.barrier()
        self.driver.zero_grad(self.set_grad_to_none)

        try:
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
                self.epoch_validate()
                self.driver.barrier()
            self.on_train_end()
            self.driver.barrier()

        except EarlyStopException as e:
            logger.info(f"Catch early stop exception: {e.msg}.")
            self.on_exception(e)
        except KeyboardInterrupt as e:
            self.driver.on_exception()
            self.on_exception(e)
            if not catch_KeyboardInterrupt:
                raise e
        except BaseException as e:
            self.driver.on_exception()
            self.on_exception(e)
            raise e

    def _set_num_eval_batch_per_dl(self, num_eval_batch_per_dl):
        def _validate_fn(trainer: Trainer, validate_fn: Callable) -> None:
            trainer.on_validate_begin()
            _validate_res: dict = validate_fn()
            trainer.on_validate_end(_validate_res)

        if self.evaluator is not None:
            self.run_evaluate = partial(_validate_fn, self, partial(self.evaluator.run, num_eval_batch_per_dl))

    def step_validate(self):
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

    def epoch_validate(self):
        """
        在每个 epoch 结束后调用，根据设置执行 evaluate 。

        :return:
        """
        if self.evaluator is not None:
            if isinstance(self.evaluate_every, int) and self.evaluate_every < 0:
                validate_every = -self.evaluate_every
                if self.cur_epoch_idx % validate_every == 0:
                    self.run_evaluate()

    def add_callback_fn(self, event: Optional[Union[Events, EventsList]], fn: Callable):
        r"""
        在初始化一个 trainer 实例后，用户可以使用这一函数来方便地添加 callback 函数；
        这一函数应当交给具体的 trainer 实例去做，因此不需要 `mark` 参数；

        :param event: 特定的 callback 时机，用户需要为该 callback 函数指定其属于哪一个 callback 时机；
        :param fn: 具体的 callback 函数；
        """
        if not isinstance(event, (_SingleEventState, EventsList)):
            raise ValueError("parameter event should only be `Events` or `EventsList` type.")

        _custom_callback = _CallbackWrapper(event, fn)
        self.callback_manager.dissect_one_callback(_custom_callback)

    @classmethod
    def on(cls, event: Optional[Union[Events, EventsList]], marker: Optional[str] = None):
        r"""
        函数修饰器，用户可以使用该函数来方便地将一个函数转变为 callback 函数，从而进行训练流程中的控制；
        注意如果你使用该函数修饰器来为你的训练添加 callback，请务必保证你加入 callback 函数的代码在实例化 `Trainer` 之前；

        :param event: 特定的 callback 时机，用户需要为该 callback 函数指定其属于哪一个 callback 时机；
        :param marker: 用来标记该 callback 函数属于哪几个具体的 trainer 实例；两个特殊情况：1.当 `marker` 为 None（默认情况）时，
         表示该 callback 函数只属于代码下方最近的一个 trainer 实例；2.当 `marker` 为 'all' 时，该 callback 函数会被所有的 trainer
         实例使用；
        :return: 返回原函数；
        """

        def wrapper(fn: Callable) -> Callable:
            cls._custom_callbacks[marker].append((event, fn))
            callback_fn_args = get_fn_arg_names(getattr(Callback, event.value))[1:]
            _check_valid_parameters_number(fn, callback_fn_args)
            return fn

        return wrapper

    def _fetch_matched_fn_callbacks(self):
        """
        因为对于使用装饰器加入的函数 callback，我们是加在类属性中，因此在初始化一个具体的 trainer 实例后，我们需要从 Trainer 的
        callback 类属性中将属于其的 callback 函数拿到，然后加入到 callback_manager 中；
        """
        _own_callbacks: List = copy.deepcopy(self._custom_callbacks["all"])
        _own_callbacks.extend(self._custom_callbacks[None])
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
            logger.warning("You have customized your 'batch_step_fn' in the 'train_batch_loop' and also use these "
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

    def load_model(self, folder: Union[str, Path, BinaryIO, io.BytesIO], only_state_dict: bool = False,
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
        # with self.get_no_sync_context():
        #     self.driver.backward(loss)
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







