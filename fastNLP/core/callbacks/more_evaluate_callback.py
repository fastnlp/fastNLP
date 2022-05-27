__all__ = [
    'MoreEvaluateCallback'
]

import os
from typing import Union, Callable, Optional, Dict

from fastNLP.core.log import logger
from .has_monitor_callback import HasMonitorCallback
from .topk_saver import TopkSaver


class MoreEvaluateCallback(HasMonitorCallback):
    """
    当评测时需要调用不同的 evaluate_fn （例如在大部分生成任务中，一般使用训练 loss 作为训练过程中的 evaluate ；但同时在训练到
    一定 epoch 数量之后，会让 model 生成的完整的数据评测 bleu 等。此刻就可能需要两种不同的 evaluate_fn ），只使用 Trainer
    无法满足需求，可以通过调用本 callback 进行。如果需要根据本 callback 中的评测结果进行模型保存，请传入 topk 以及
    topk_monitor 等相关参数。可以通过 evaluate_every 或 watch_monitor 控制触发进行 evaluate 的条件。

    如果设置了 evaluate 结果更好就保存的话，将按如下文件结构进行保存::

        - folder/
            - YYYY-mm-dd-HH_MM_SS_fffff/  # 自动根据当前脚本的启动时间创建的
                - {save_object}-epoch_{epoch_idx}-batch_{global_batch_idx}-{topk_monitor}_{monitor_value}/  # 满足topk条件存储文件名

    :param dataloaders: 需要评估的数据
    :param metrics: 使用的 metrics 。
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
    :param watch_monitor: 这个值用来表示监控的 Trainer 中的 evaluate 结果的，当该值不为 None ，evaluate_every 失效。本参数的
        意义是，当检测到 Trainer 中 evaluate results 的 {watch_monitor} 的结果更好时，则进行一次 evaluate 。该参数有两种
        取值: (1) str 类型，监控的 metric 值。如果在 evaluation 结果中没有找到完全一致的名称，将使用 最长公共字符串算法 找到最
        匹配的那个作为 monitor ; (2) 也可以传入一个函数，接受参数为 evaluation 的结果(字典类型)，返回一个 float 值作为 monitor
        的结果，如果当前结果中没有相关的monitor 值请返回 None 。
    :param watch_monitor_larger_better: watch_monitor 是否越大越好。
    :param evaluate_fn: 用来控制 `Evaluator` 在评测的前向传播过程中是调用哪一个函数，例如是 `model.evaluate_step` 还是
        `model.forward`；(1) 如果该值是 None，那么我们会默认使用 `evaluate_step` 当做前向传播的函数，如果在模型中没有
        找到该方法，则使用 `model.forward` 函数；(2) 如果为 str 类型，则尝试从 model 中寻找该方法，找不到则报错。
    :param num_eval_sanity_batch: 在初始化 Evaluator 后运行多少个 sanity check 的 batch ，检测一下。
    :param topk: 如果需要根据当前 callback 中的 evaluate 结果保存模型或 Trainer ，可以通过设置 tokp 实现。（1）为 -1 表示每次
        evaluate 后都保存；（2）为 0 （默认），表示不保存；（3）为整数，表示保存性能最 topk 个。
    :param topk_monitor: 如果需要根据当前 callback 中的 evaluate 结果保存。这个参数是指在当前 callback 中的 evaluate 结果寻找
    :param topk_larger_better: topk_monitor 的值是否时越大越好。
    :param folder: 保存的文件夹，fastNLP 将在该文件下以时间戳创建子文件夹，并在里面保存。因此不同次运行可以将被保存到不同的
        时间戳文件夹中。如果为 None ，默认使用当前文件夹。
    :param only_state_dict: 保存模型时是否只保存 state_dict 。当 model_save_fn 不为 None 时，该参数无效。
    :param save_object: 可选 ['trainer', 'model']，表示在保存时的保存对象为 ``trainer+model`` 还是 只是 ``model`` 。如果
        保存 ``trainer`` 对象的话，将会保存 :class:~fastNLP.Trainer 的相关状态，可以通过 :meth:`Trainer.load_checkpoint` 加载该断
        点继续训练。如果保存的是 ``Model`` 对象，则可以通过 :meth:`Trainer.load_model` 加载该模型权重。
    :param model_save_fn: 个性化的保存函数，当触发保存操作时，就调用这个函数，这个函数应当接受一个文件夹作为参数，不返回任何东西。
        如果传入了 model_save_fn 函数，fastNLP 将不再进行模型相关的保存。在多卡场景下，我们只在 rank 0 上会运行该函数。
    :param save_evaluate_results: 是否保存 evaluate 的结果。如果为 True ，在保存 topk 模型的 folder 中还将额外保存一个
         ``fastnlp_evaluate_results.json`` 文件，记录当前的 results。仅在设置了 topk 的场景下有用，默认为 True 。
    :param save_kwargs: dict。更多的保存相关的参数。
    :param kwargs: 其它与 Evaluator 相关的初始化参数，如果不传入，将从 Trainer 中获取。
    """
    def __init__(self, dataloaders, metrics:Dict, evaluate_every:Optional[Union[int, Callable]]=-1,
                 watch_monitor:Union[str, Callable]=None, watch_monitor_larger_better:bool=True,
                 evaluate_fn=None, num_eval_sanity_batch=2,
                 topk=0, topk_monitor=None, topk_larger_better=True,
                 folder=None, only_state_dict=True, save_object='model', model_save_fn=None,
                 save_evaluate_results=True, save_kwargs=None,
                 **kwargs):
        super(MoreEvaluateCallback, self).__init__(watch_monitor, watch_monitor_larger_better,
                                               must_have_monitor=False)

        if watch_monitor is None and evaluate_every is None:
            raise RuntimeError("`evaluate_every` and `watch_monitor` cannot be None at the same time.")
        if watch_monitor is not None and evaluate_every is not None:
            raise RuntimeError(f"`evaluate_every`({evaluate_every}) and `watch_monitor`({watch_monitor}) "
                               f"cannot be set at the same time.")

        if topk_monitor is not None and topk == 0:
            raise RuntimeError("`topk_monitor` is set, but `topk` is 0.")
        if topk != 0 and topk_monitor is None:
            raise RuntimeError("`topk` is set, but `topk_monitor` is None.")
        assert save_object in ['trainer', 'model']

        self.dataloaders = dataloaders
        self.metrics = metrics
        self.evaluate_every = evaluate_every
        self.evaluate_fn = evaluate_fn
        self.num_eval_sanity_batch = num_eval_sanity_batch
        if save_kwargs is None:
            save_kwargs = {}
        self.topk_saver = TopkSaver(topk=topk, monitor=topk_monitor, larger_better=topk_larger_better,
                                    folder=folder, only_state_dict=only_state_dict,
                                    model_save_fn=model_save_fn, save_evaluate_results=save_evaluate_results,
                                    save_object=save_object, **save_kwargs)
        self.kwargs = kwargs

    @property
    def need_reproducible_sampler(self) -> bool:
        return self.topk_saver.save_object == 'trainer'

    def on_after_trainer_initialized(self, trainer, driver):
        # 如果是需要 watch 的，不能没有 evaluator
        if self.monitor is not None:
            assert trainer.evaluator is not None, f"You set `watch_monitor={self.monitor}`, but no " \
                                                  f"evaluate_dataloaders is provided in Trainer."

        if trainer.evaluate_fn is self.evaluate_fn:
            logger.warning_once("The `evaluate_fn` is the same as in Trainer, there seems no need to use "
                                "`MoreEvaluateCallback`.")

        # 初始化 evaluator , 同时避免调用 super 对 monitor 赋值
        kwargs = {
            'model': self.kwargs.get('model', trainer.model),
            'dataloaders': self.dataloaders,
            'metrics': self.metrics,
            'driver': self.kwargs.get('driver', trainer.driver),
            'device': self.kwargs.get('device', trainer.device),
            'evaluate_batch_step_fn': self.kwargs.get('evaluate_batch_step_fn', trainer.evaluate_batch_step_fn),
            'evaluate_fn': self.evaluate_fn,
            'input_mapping': self.kwargs.get('input_mapping', trainer.input_mapping),
            'output_mapping': self.kwargs.get('output_mapping', trainer.output_mapping),
            'fp16': self.kwargs.get('fp16', trainer.fp16),
            'use_dist_sampler': self.kwargs.get('use_dist_sampler',
                                                trainer.kwargs.get('eval_use_dist_sampler', None)),
            'progress_bar': self.kwargs.get('progress_bar', trainer.kwargs.get('progress_bar', 'auto')),
            'verbose': self.kwargs.get('verbose', 1)
        }

        for key, value in self.kwargs.items():
            if key not in kwargs:
                kwargs[key] = value
        from fastNLP.core.controllers.evaluator import Evaluator
        self.evaluator = Evaluator(**kwargs)
        if self.num_eval_sanity_batch>0:
            results = self.evaluator.run(num_eval_batch_per_dl=self.num_eval_sanity_batch)
            self.topk_saver.get_monitor_value(results)

    def on_evaluate_end(self, trainer, results):
        if self.is_better_results(results, keep_if_better=True):
            results = self.evaluator.run()
            self.topk_saver.save_topk(trainer, results)

    def on_train_epoch_end(self, trainer):
        if self.monitor is not None:
            return
        if isinstance(self.evaluate_every, int) and self.evaluate_every < 0:
            evaluate_every = -self.evaluate_every
            if trainer.cur_epoch_idx % evaluate_every == 0:
                results = self.evaluator.run()
                self.topk_saver.save_topk(trainer, results)

    def on_train_batch_end(self, trainer):
        if self.monitor is not None:
            return
        if callable(self.evaluate_every):
            if self.evaluate_every(trainer):
                results = self.evaluator.run()
                self.topk_saver.save_topk(trainer, results)
        elif self.evaluate_every > 0 and trainer.global_forward_batches % self.evaluate_every == 0:
            results = self.evaluator.run()
            self.topk_saver.save_topk(trainer, results)

    def on_save_checkpoint(self, trainer) -> Dict:
        states = {'topk_saver': self.topk_saver.state_dict()}
        if isinstance(self._real_monitor, str):
            states['_real_monitor'] = self._real_monitor
            states['monitor_value'] = self.monitor_value
        return states

    def on_load_checkpoint(self, trainer, states: Optional[Dict]):
        topk_saver_states = states['topk_saver']
        self.topk_saver.load_state_dict(topk_saver_states)
        if '_real_monitor' in states:
            self._real_monitor = states["_real_monitor"]
            self.monitor_value = states['monitor_value']

    @property
    def callback_name(self):
        metric_names = '+'.join(sorted(self.metrics.keys()))
        return f'more_evaluate_callback#metric_name-{metric_names}#monitor-{self.monitor_name}#topk_saver:{self.topk_saver}'

