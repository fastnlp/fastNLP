r"""
callback模块实现了 fastNLP 中的许多 callback 类，用于增强 :class:`~fastNLP.Trainer` 类。

虽然Trainer本身已经集成了一些功能，但仍然不足以囊括训练过程中可能需要到的功能，
比如负采样，learning rate decay 和 early stop等。
为了解决这个问题，fastNLP引入了callback的机制，:class:`~fastNLP.Callback` 是一种在Trainer训练过程中特定阶段会运行的函数集合。
关于 :class:`~fastNLP.Trainer` 的详细文档，请参见 :mod:`trainer 模块<fastNLP.core.trainer>`

我们将 :meth:`~fastNLP.Trainer.train` 这个函数内部分为以下的阶段，在对应阶段会触发相应的调用::

    callback.on_train_begin()  # 开始进行训练
    for i in range(1, n_epochs+1):
        callback.on_epoch_begin()  # 开始新的epoch
        for batch_x, batch_y in Batch:
            callback.on_batch_begin(batch_x, batch_y, indices) # batch_x是设置为input的field，batch_y是设置为target的field
            获取模型输出
            callback.on_loss_begin()
            计算loss
            callback.on_backward_begin() # 可以进行一些检查，比如loss是否为None
            反向梯度回传
            callback.on_backward_end() # 进行梯度截断等
            进行参数更新
            callback.on_step_end()
            callback.on_batch_end()
            # 根据设置进行evaluation，比如这是本epoch最后一个batch或者达到一定step
            if do evaluation:
                callback.on_valid_begin()
                进行dev data上的验证
                callback.on_valid_end()  # 可以进行在其它数据集上进行验证
        callback.on_epoch_end()  # epoch结束调用
    callback.on_train_end() # 训练结束
    callback.on_exception() # 这是一个特殊的步骤，在训练过程中遭遇exception会跳转到这里。

如下面的例子所示，我们可以使用内置的 callback 组件，或者继承 :class:`~fastNLP.core.callback.Callback`
定义自己的 callback 组件::
    
    from fastNLP import Callback, EarlyStopCallback, Trainer, CrossEntropyLoss, AccuracyMetric
    from fastNLP.models import CNNText
    
    start_time = time.time()
    
    class MyCallback(Callback):
        def on_epoch_end(self):
            print('{:d}ms\n\n'.format(round((time.time()-start_time)*1000)))
    
    model = CNNText((len(vocab),50), num_classes=5, padding=2, dropout=0.1)
    trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data, loss=CrossEntropyLoss(),
                      metrics=AccuracyMetric(), callbacks=[MyCallback(),EarlyStopCallback(10)])
    trainer.train()

"""
__all__ = [
    "Callback",

    "GradientClipCallback",
    "EarlyStopCallback",
    "FitlogCallback",
    "EvaluateCallback",
    "LRScheduler",
    "ControlC",
    "LRFinder",
    "TensorboardCallback",
    "WarmupCallback",
    "SaveModelCallback",
    
    "CallbackException",
    "EarlyStopError",
    "CheckPointCallback"
]

import os
import sys
from copy import deepcopy

import torch

from .utils import _save_model

try:
    from tensorboardX import SummaryWriter
    
    tensorboardX_flag = True
except:
    tensorboardX_flag = False

from .dataset import DataSet
from .tester import Tester
from ._logger import logger
from .utils import _check_fp16
from ._parallel_utils import _model_contains_inner_module

try:
    import fitlog
except:
    pass

try:
    from apex import amp
except:
    amp = None


class Callback(object):
    r"""
    Callback是fastNLP中被设计用于增强 :class:`~fastNLP.Trainer` 的类。
    如果Callback被传递给了 Trainer , 则 Trainer 会在对应的阶段调用Callback的函数，
    具体调用时机可以通过 :mod:`trainer 模块<fastNLP.core.trainer>` 查看。
    这是Callback的基类，所有的callback必须继承自这个类

    """
    
    def __init__(self):
        super(Callback, self).__init__()
        self._trainer = None  # 在Trainer内部被重新赋值
        self._disabled = False

    def __repr__(self):
        return self.__class__.__name__

    @property
    def trainer(self):
        r"""
        该属性可以通过self.trainer获取到，一般情况下不需要使用这个属性。
        """
        return self._trainer
    
    @property
    def step(self):
        r"""当前运行到的step, 范围为[1, self.n_steps+1)"""
        return self._trainer.step
    
    @property
    def n_steps(self):
        r"""Trainer一共会采多少个batch。当Trainer中update_every设置为非1的值时，该值不等于update的次数"""
        return self._trainer.n_steps
    
    @property
    def batch_size(self):
        r"""train和evaluate时的batch_size为多大"""
        return self._trainer.batch_size
    
    @property
    def epoch(self):
        r"""当前运行的epoch数，范围是[1, self.n_epochs+1)"""
        return self._trainer.epoch
    
    @property
    def n_epochs(self):
        r"""一共会运行多少个epoch"""
        return self._trainer.n_epochs
    
    @property
    def optimizer(self):
        r"""初始化Trainer时传递的Optimizer"""
        return self._trainer.optimizer
    
    @property
    def model(self):
        r"""正在被Trainer训练的模型"""
        return self._trainer.model
    
    @property
    def pbar(self):
        r"""如果在Callback中需要打印内容，请使用self.pbar.write(str)。否则可能出现命令行显示效果不太好的问题。在
        on_train_begin(), on_train_end(), on_exception()中请不要使用该属性，通过print输出即可。"""
        return self._trainer.pbar
    
    @property
    def update_every(self):
        r"""Trainer中的模型多少次反向传播才进行一次梯度更新，在Trainer初始化时传入的。"""
        return self._trainer.update_every
    
    @property
    def batch_per_epoch(self):
        r"""每个epoch一共有多少个batch，只有在on_epoch_begin之后才能调用该属性。"""
        return self._trainer.batch_per_epoch

    @property
    def is_master(self):
        return self._trainer.is_master

    @property
    def disabled(self):
        return self._disabled

    @property
    def logger(self):
        return getattr(self._trainer, 'logger', logger)

    def on_train_begin(self):
        r"""
        在Train过程开始之前调用。

        :return:
        """
        pass
    
    def on_epoch_begin(self):
        r"""
        在每个epoch开始之前调用一次

        :return:
        """
        pass
    
    def on_batch_begin(self, batch_x, batch_y, indices):
        r"""
        每次采集到一个batch的数据则调用一次。这里对batch_x或batch_y删除添加内容是可以影响到Trainer中内容的。所以在这一步
        可以进行一些负采样之类的操作

        :param dict batch_x: DataSet中被设置为input的field的batch。
        :param dict batch_y: DataSet中被设置为target的field的batch。
        :param list(int) indices: 这次采样使用到的indices，可以通过DataSet[indices]获取出这个batch采出的Instance，在一些
            情况下可以帮助定位是哪个Sample导致了错误。仅当num_workers=0时有效。
        :return:
        """
        pass
    
    def on_loss_begin(self, batch_y, predict_y):
        r"""
        在计算loss前调用，即这里修改batch_y或predict_y的值是可以影响到loss计算的。

        :param dict batch_y: 在DataSet中被设置为target的field的batch集合。
        :param dict predict_y: 模型的forward()返回的结果。
        :return:
        """
        pass
    
    def on_backward_begin(self, loss):
        r"""
        在loss得到之后，但在反向传播之前。可能可以进行loss是否为NaN的检查。

        :param torch.Tensor loss: 计算得到的loss值
        :return:
        """
        pass
    
    def on_backward_end(self):
        r"""
        反向梯度传播已完成，但由于update_every的设置，可能并不是每一次调用都有梯度。到这一步，还没有更新参数。

        :return:
        """
        pass
    
    def on_step_end(self):
        r"""
        到这里模型的参数已经按照梯度更新。但可能受update_every影响，并不是每次都更新了。

        :return:
        """
        pass
    
    def on_batch_end(self):
        r"""
        这一步与on_step_end是紧接着的。只是为了对称性加上了这一步。

        """
        pass
    
    def on_valid_begin(self):
        r"""
        如果Trainer中设置了验证，则发生验证前会调用该函数

        :return:
        """
        pass
    
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        r"""
        每次执行验证集的evaluation后会调用。

        :param Dict[str: Dict[str: float]] eval_result: , evaluation的结果。一个例子为{'AccuracyMetric':{'acc':1.0}}，即
            传入的dict是有两层，第一层是metric的名称，第二层是metric的具体指标。
        :param str metric_key: 初始化Trainer时传入的metric_key。
        :param torch.Optimizer optimizer: Trainer中使用的优化器。
        :param bool is_better_eval: 当前dev结果是否比之前的好。
        :return:
        """
        pass
    
    def on_epoch_end(self):
        r"""
        每个epoch结束将会调用该方法
        """
        pass
    
    def on_train_end(self):
        r"""
        训练结束，调用该方法
        """
        pass
    
    def on_exception(self, exception):
        r"""
        当训练过程出现异常，会触发该方法
        :param exception: 某种类型的Exception，比如KeyboardInterrupt等
        """
        pass


def _transfer(func):
    r"""装饰器，将对CallbackManager的调用转发到各个Callback子类.
    
    :param func:
    :return:
    """
    
    def wrapper(manager, *arg):
        returns = []
        for callback in manager.callbacks:
            if callback.disabled:
                continue
            returns.append(getattr(callback, func.__name__)(*arg))
        return returns
    
    return wrapper


class CallbackManager(Callback):
    r"""
    内部使用的Callback管理类
    """
    def __init__(self, env, callbacks=None):
        r"""

        :param dict env: The key is the name of the Trainer attribute(str). The value is the attribute itself.
        :param List[Callback] callbacks:
        """
        super(CallbackManager, self).__init__()
        # set attribute of trainer environment
        self._env = env
        self.callbacks = []
        if callbacks:
            self.callbacks = self.prepare_callbacks(callbacks)

    def prepare_callbacks(self, callbacks):
        if not callbacks:
            return []
        if isinstance(callbacks, list):
            if all([isinstance(cb, Callback) for cb in callbacks]) is True:
                pass
            else:
                obj = [not isinstance(cb, Callback) for cb in callbacks][0]
                raise TypeError(f"Expect sub-classes of Callback. Got {type(obj)}")
        else:
            raise TypeError(f"Expect callbacks in CallbackManager(callbacks) to be list. Got {type(callbacks)}.")

        for env_name, env_val in self._env.items():
            for callback in callbacks:
                setattr(callback, '_' + env_name, env_val)  # Callback.trainer
        return callbacks

    @_transfer
    def on_train_begin(self):
        pass
    
    @_transfer
    def on_epoch_begin(self):
        pass
    
    @_transfer
    def on_batch_begin(self, batch_x, batch_y, indices):
        pass
    
    @_transfer
    def on_loss_begin(self, batch_y, predict_y):
        pass
    
    @_transfer
    def on_backward_begin(self, loss):
        pass
    
    @_transfer
    def on_backward_end(self):
        pass
    
    @_transfer
    def on_step_end(self):
        pass
    
    @_transfer
    def on_batch_end(self):
        pass
    
    @_transfer
    def on_valid_begin(self):
        pass
    
    @_transfer
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        pass

    @_transfer
    def on_validation(self):
        pass
    
    @_transfer
    def on_epoch_end(self):
        pass
    
    @_transfer
    def on_train_end(self):
        pass
    
    @_transfer
    def on_exception(self, exception):
        pass


class DistCallbackManager(CallbackManager):
    def __init__(self, env, callbacks_all=None, callbacks_master=None):
        super(DistCallbackManager, self).__init__(env)
        assert 'trainer' in env
        self._trainer = env['trainer']
        self.callbacks_master = []
        self.callbacks_all = []
        self.add_callback(callbacks_all, master=False)
        self.add_callback(callbacks_master, master=True)

    def patch_callback(self, callbacks, disabled):
        if not callbacks:
            return
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]
        for cb in callbacks:
            cb._disabled = disabled

    def add_callback(self, cb, master=False):
        if master:
            self.patch_callback(cb, not self.is_master)
            self.callbacks_master += self.prepare_callbacks(cb)
        else:
            self.callbacks_all += self.prepare_callbacks(cb)
        self.callbacks = self.callbacks_all + self.callbacks_master


class GradientClipCallback(Callback):
    r"""
    每次backward前，将parameter的gradient clip到某个范围。
    """
    
    def __init__(self, parameters=None, clip_value=1, clip_type='norm'):
        r"""
        
        :param None,torch.Tensor,List[torch.Tensor] parameters: 一般通过model.parameters()获得。
            如果为None则默认对Trainer的model中所有参数进行clip
        :param float clip_value: 将gradient 限制到[-clip_value, clip_value]。clip_value应该为正数
        :param str clip_type: 支持'norm', 'value'
            两种::
    
                1 'norm', 将gradient的norm rescale到[-clip_value, clip_value]
            
                2 'value', 将gradient限制在[-clip_value, clip_value],
                    小于-clip_value的gradient被赋值为-clip_value;
                    大于clip_value的gradient被赋值为clip_value.
        """
        super().__init__()
        
        from torch import nn
        if clip_type == 'norm':
            self.clip_fun = nn.utils.clip_grad_norm_
        elif clip_type == 'value':
            self.clip_fun = nn.utils.clip_grad_value_
        else:
            raise ValueError("Only supports `norm` or `value` right now.")
        if parameters is not None:
            self.parameters = list(parameters)
        else:
            self.parameters = None
        self.clip_value = clip_value
    
    def on_backward_end(self):
        if self.step%self.update_every==0:
            if self.parameters is None:
                if getattr(self.trainer, 'fp16', ''):
                    _check_fp16()
                    self.clip_fun(amp.master_params(self.optimizer), self.clip_value)
                else:
                    self.clip_fun(self.model.parameters(), self.clip_value)
            else:
                self.clip_fun(self.parameters, self.clip_value)


class EarlyStopCallback(Callback):
    r"""
    多少个epoch没有变好就停止训练，相关类 :class:`~fastNLP.core.callback.EarlyStopError`
    """
    
    def __init__(self, patience):
        r"""
        
        :param int patience: epoch的数量
        """
        super(EarlyStopCallback, self).__init__()
        self.patience = patience
        self.wait = 0
    
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if not is_better_eval:
            # current result is getting worse
            if self.wait == self.patience:
                raise EarlyStopError("Early stopping raised.")
            else:
                self.wait += 1
        else:
            self.wait = 0
    
    def on_exception(self, exception):
        if isinstance(exception, EarlyStopError):
            logger.info("Early Stopping triggered in epoch {}!".format(self.epoch))
        else:
            raise exception  # 抛出陌生Error


class FitlogCallback(Callback):
    r"""
    该callback可将loss和progress写入到fitlog中; 如果Trainer有dev的数据，将自动把dev的结果写入到log中; 同时还支持传入
    一个(或多个)test数据集进行测试(只有在trainer具有dev时才能使用)，每次在dev上evaluate之后会在这些数据集上验证一下。
    并将验证结果写入到fitlog中。这些数据集的结果是根据dev上最好的结果报道的，即如果dev在第3个epoch取得了最佳，则
    fitlog中记录的关于这些数据集的结果就是来自第三个epoch的结果。
    """

    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=1, log_exception=False):
        r"""
        
        :param ~fastNLP.DataSet,Dict[~fastNLP.DataSet] data: 传入DataSet对象，会使用多个Trainer中的metric对数据进行验证。如果需要
            传入多个DataSet请通过dict的方式传入，dict的key将作为对应dataset的name传递给fitlog。data的结果的名称以'data'开头。
        :param ~fastNLP.Tester,Dict[~fastNLP.Tester] tester: Tester对象，将在on_valid_end时调用。tester的结果的名称以'tester'开头
        :param int log_loss_every: 多少个step记录一次loss(记录的是这几个batch的loss平均值)，如果数据集较大建议将该值设置得
            大一些，不然会导致log文件巨大。默认为0, 即不要记录loss。
        :param int verbose: 是否在终端打印evaluation的结果，0不打印。
        :param bool log_exception: fitlog是否记录发生的exception信息
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self._log_exception = log_exception
        assert isinstance(log_loss_every, int) and log_loss_every>=0
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")
        
        self.verbose = verbose
        self._log_loss_every = log_loss_every
        self._avg_loss = 0

    def on_train_begin(self):
        if (len(self.datasets) > 0 or len(self.testers) > 0) and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra data to do evaluation.")
        
        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.batch_size),
                                metrics=self.trainer.metrics,
                                verbose=0,
                                use_tqdm=self.trainer.test_use_tqdm,
                                sampler=self.trainer.kwargs.get('test_sampler', None))
                self.testers[key] = tester
        fitlog.add_progress(total_steps=self.n_steps)
    
    def on_backward_begin(self, loss):
        if self._log_loss_every>0:
            self._avg_loss += loss.item()
            if self.step%self._log_loss_every==0:
                fitlog.add_loss(self._avg_loss/self._log_loss_every*self.update_every, name='loss', step=self.step, epoch=self.epoch)
                self._avg_loss = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        if len(self.testers) > 0:
            for key, tester in self.testers.items():
                try:
                    eval_result = tester.test()
                    if self.verbose != 0:
                        self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                        self.pbar.write(tester._format_eval_results(eval_result))
                    fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                    if better_result:
                        fitlog.add_best_metric(eval_result, name=key)
                except Exception as e:
                    self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
                    raise e

    def on_train_end(self):
        fitlog.finish()
    
    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')


class EvaluateCallback(Callback):
    r"""
    通过使用该Callback可以使得Trainer在evaluate dev之外还可以evaluate其它数据集，比如测试集。每一次验证dev之前都会先验证EvaluateCallback
    中的数据。
    """

    def __init__(self, data=None, tester=None):
        r"""
        :param ~fastNLP.DataSet,Dict[~fastNLP.DataSet] data: 传入DataSet对象，会使用Trainer中的metric对数据进行验证。如果需要传入多个
            DataSet请通过dict的方式传入。
        :param ~fastNLP.Tester,Dict[~fastNLP.DataSet] tester: Tester对象, 通过使用Tester对象，可以使得验证的metric与Trainer中
            的metric不一样。
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

    def on_train_begin(self):
        if len(self.datasets) > 0 and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra DataSet to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.batch_size),
                                metrics=self.trainer.metrics, verbose=0,
                                use_tqdm=self.trainer.test_use_tqdm)
                self.testers[key] = tester

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if len(self.testers) > 0:
            for key, tester in self.testers.items():
                try:
                    eval_result = tester.test()
                    self.logger.info("EvaluateCallback evaluation on {}:".format(key))
                    self.logger.info(tester._format_eval_results(eval_result))
                except Exception as e:
                    self.logger.error("Exception happens when evaluate on DataSet named `{}`.".format(key))
                    raise e

class LRScheduler(Callback):
    r"""
    对PyTorch LR Scheduler的包装以使得其可以被Trainer所使用
    """
    
    def __init__(self, lr_scheduler):
        r"""
        :param torch.optim.lr_scheduler._LRScheduler lr_scheduler: PyTorch的lr_scheduler
        """
        super(LRScheduler, self).__init__()
        import torch.optim
        if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            self.scheduler = lr_scheduler
        else:
            raise ValueError(f"Expect torch.optim.lr_scheduler for LRScheduler. Got {type(lr_scheduler)}.")
    
    def on_epoch_end(self):
        self.scheduler.step(self.epoch)


class ControlC(Callback):
    r"""
    检测到 control+C 时的反馈
    """
    
    @staticmethod
    def quit_all():
        import sys
        sys.exit(0)  # 直接退出程序
    
    def __init__(self, quit_and_do, action=quit_all):
        r"""
        :param bool quit_and_do: 若为True,则检测到control+C 进行后续操作（默认值为：直接退出程序）；否则只退出Trainer。
        """
        
        super(ControlC, self).__init__()
        if type(quit_and_do) != bool:
            raise ValueError("In KeyBoardInterrupt, quit_and_do arguemnt must be a bool.")
        self.quit_and_do = quit_and_do
        self.action = action
    
    def on_exception(self, exception):
        if isinstance(exception, KeyboardInterrupt):
            if self.quit_and_do is True:
                self.action()
            else:
                pass
        else:
            raise exception  # 抛出陌生Error


class SmoothValue(object):
    r"""work for LRFinder"""
    
    def __init__(self, beta: float):
        self.beta, self.n, self.mov_avg = beta, 0, 0
        self.smooth = None
    
    def add_value(self, val: float) -> None:
        r"""Add `val` to calculate updated smoothed value."""
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


class LRFinder(Callback):
    r"""
    用第一个 epoch 找最佳的学习率，从第二个epoch开始应用它
    """
    
    def __init__(self, start_lr=1e-6, end_lr=10):
        r"""
        
        :param float start_lr: 学习率下界
        :param float end_lr: 学习率上界
        """
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        
        self.stop = False
        self.best_loss = 0.
        self.best_lr = None
        self.loss_history = []
        self.smooth_value = SmoothValue(0.8)
        self.opt = None
        self.find = None

    @property
    def lr_gen(self):
        scale = (self.end_lr - self.start_lr) / self.batch_per_epoch
        return (self.start_lr + scale * (step + 1) for step in range(self.batch_per_epoch))
    
    @property
    def num_it(self):
        return self.batch_per_epoch
    
    def on_epoch_begin(self):
        if self.epoch == 1:  # first epoch
            self.opt = self.trainer.optimizer  # pytorch optimizer
            self.opt.param_groups[0]["lr"] = self.start_lr
            # save model
            torch.save(self.model.state_dict(), 'tmp')
            self.find = True
    
    def on_backward_begin(self, loss):
        if self.find:
            if torch.isnan(loss) or self.stop is True:
                self.stop = True
                return
            loss_val = loss.detach().mean().item()
            self.loss_history.append(loss_val)
            self.smooth_value.add_value(loss_val)
            if self.best_loss == 0. or self.smooth_value.smooth < self.best_loss:
                self.best_loss = self.smooth_value.smooth
                self.best_lr = self.opt.param_groups[0]["lr"]
    
    def on_batch_end(self, *args):
        if self.find:
            lr = next(self.lr_gen, None)
            if lr is None or self.stop is True or self.loss_history[-1] > 4 * self.best_loss:
                self.stop = True
                return
            self.opt.param_groups[0]["lr"] = lr
            # self.loader.load_pytorch(self.trainer.model, "tmp")
    
    def on_epoch_end(self):
        if self.epoch == 1:  # first epoch
            self.opt.param_groups[0]["lr"] = self.best_lr
            self.find = False
            # reset model
            states = torch.load('tmp')
            self.model.load_state_dict(states)
            os.remove('tmp')
            self.pbar.write("Model reset. \nFind best lr={}".format(self.best_lr))


class TensorboardCallback(Callback):
    r"""
    接受以下一个或多个字符串作为参数：
    - "model"
    - "loss"
    - "metric"
    
    .. warning::
        fastNLP 已停止对此功能的维护，请等待 fastNLP 兼容 PyTorch1.1 的下一个版本。
        或者使用和 fastNLP 高度配合的 fitlog（参见 :doc:`/tutorials/extend_3_fitlog` ）。
        
    """
    
    def __init__(self, *options):
        super(TensorboardCallback, self).__init__()
        args = {"model", "loss", "metric"}
        for opt in options:
            if opt not in args:
                raise ValueError("Unrecognized argument {}. Expect one of {}".format(opt, args))
        self.options = options
        self._summary_writer = None
        self.graph_added = False
    
    def on_train_begin(self):
        save_dir = self.trainer.save_path
        if save_dir is None:
            path = os.path.join("./", 'tensorboard_logs_{}'.format(self.trainer.start_time))
        else:
            path = os.path.join(save_dir, 'tensorboard_logs_{}'.format(self.trainer.start_time))
        if tensorboardX_flag:
            self._summary_writer = SummaryWriter(path)
        else:
            self._summary_writer = None
    
    def on_batch_begin(self, batch_x, batch_y, indices):
        if "model" in self.options and self.graph_added is False:
            # tesorboardX 这里有大bug，暂时没法画模型图
            # from fastNLP.core.utils import _build_args
            # inputs = _build_args(self.trainer.model, **batch_x)
            # args = tuple([value for value in inputs.values()])
            # args = args[0] if len(args) == 1 else args
            # self._summary_writer.add_graph(self.trainer.model, torch.zeros(32, 2))
            self.graph_added = True
    
    def on_backward_begin(self, loss):
        if "loss" in self.options and self._summary_writer:
            self._summary_writer.add_scalar("loss", loss.item(), global_step=self.trainer.step)
        
        if "model" in self.options and self._summary_writer:
            for name, param in self.trainer.model.named_parameters():
                if param.requires_grad:
                    self._summary_writer.add_scalar(name + "_mean", param.mean(), global_step=self.trainer.step)
                    # self._summary_writer.add_scalar(name + "_std", param.std(), global_step=self.trainer.step)
                    self._summary_writer.add_scalar(name + "_grad_mean", param.grad.mean(),
                                                    global_step=self.trainer.step)
    
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if "metric" in self.options and self._summary_writer:
            for name, metric in eval_result.items():
                for metric_key, metric_val in metric.items():
                    self._summary_writer.add_scalar("valid_{}_{}".format(name, metric_key), metric_val,
                                                    global_step=self.trainer.step)
    
    def on_train_end(self):
        if self._summary_writer:
            self._summary_writer.close()
            del self._summary_writer
    
    def on_exception(self, exception):
        if hasattr(self, "_summary_writer"):
            self._summary_writer.close()
            del self._summary_writer


class CheckPointCallback(Callback):
    def __init__(self, save_path, delete_when_train_finish=True, recovery_fitlog=True):
        r"""
        用于在每个epoch结束的时候保存一下当前的Trainer状态，可以用于恢复之前的运行。使用最近的一个epoch继续训练
        一段示例代码
        Example1::

            >>> callback = CheckPointCallback('chkp.pt')
            >>> trainer = Trainer(xxx, callback=callback)
            >>> trainer.train()  # 如果训练过程没结束就fail，请直接再次运行即可（请务必保证与上次使用了完全相同的数据与超参数）

        Example2::

            >>> fitlog.set_log_dir('xxx')
            >>> callback = CheckPointCallback('chkp.pt')  # 一定要在set_log_dir下一行就接着CheckPointCallback
            >>> trainer = Trainer(xxx, callback=callback)
            >>> trainer.train()  # 如果训练过程没结束就fail，请直接再次运行即可（请务必保证与上次使用了完全相同的数据与超参数）

        :param str save_path: 将状态保存到哪个位置。需要指定一个具体的路径，比如'checkpoints/chtp.pt'。如果检查到该文件存在，将在
            Trainer开始训练的时候自动从这个Checkpoint处开始运行。
        :param bool delete_when_train_finish: 如果Train正常运行完毕，是否自动删除。删除该文件可以使得路径自动复用。
        :param bool recovery_fitlog: 是否恢复fitlog为对应的log，如果为True请将本Callback放在fitlog.set_log_dir后面一行初始化。
            如果为False，将新建一个log folder否则继续使用之前的。
        """
        super().__init__()
        self.save_path = os.path.abspath(os.path.expanduser(save_path))
        self.delete_when_train_finish = delete_when_train_finish
        self.recover_fitlog = recovery_fitlog
        try:
            import fitlog
        except:
            self.recover_fitlog = False
        if os.path.exists(os.path.expanduser(self.save_path)):
            logger.info("The train will start from the checkpoint saved in {}.".format(self.save_path))
            if self.recover_fitlog:
                states = torch.load(self.save_path)
                if 'fitlog_log_dir' in states:
                    try:
                        import fitlog
                        log_dir = states['fitlog_log_dir']
                        if 'fitlog_save_log_dir' in states:
                            log_dir = states['fitlog_save_log_dir']
                        fitlog.set_log_dir(log_dir, new_log=True)
                    except:
                        logger.error("Fail to recovery the fitlog states.")

    def on_train_begin(self):
        r"""
        当train开始时，且需要恢复上次训练时，会做以下的操作
            (1) 重新加载model权重
            (2) 重新加载optimizer的状态
            (3) 加载当前epoch数
            (4) 加载当前最佳evaluate的性能
            (5) (optional) 自动将fitlog设置到上次log出继续

        :return:
        """
        if os.path.exists(os.path.expanduser(self.save_path)):
            states = torch.load(self.save_path)
            model = self.model
            if _model_contains_inner_module(model):
                model = model.module
            model.load_state_dict(states['model'])
            self.optimizer.load_state_dict(states['optimizer'])
            self.trainer.epoch = states['epoch'] + 1 # 因为是结束储存的，所以需要从下一个epoch开始
            self.trainer.step = states['step']
            if 'best_dev_epoch' in states:
                self.trainer.best_dev_perf = states['best_dev_perf']
                self.trainer.best_dev_epoch = states['best_dev_epoch']
                self.trainer.best_dev_step = states['best_dev_step']
                self.trainer.best_metric_indicator = states['best_metric_indicator']
            logger.info("Load checkpoint from {}".format(os.path.expanduser(self.save_path)))

    def on_epoch_end(self):
        r"""
        保存状态，使得结果可以被恢复

        :param self:
        :return:
        """
        states = {}
        model = self.model
        if _model_contains_inner_module(model):
            model = model.module
        states['model'] = {name:param.cpu() for name, param in model.state_dict().items()}
        states['optimizer'] = self.optimizer.state_dict()
        states['epoch'] = self.epoch
        states['step'] = self.step
        if self.trainer.best_dev_epoch is not None:
            states['best_dev_epoch'] = self.trainer.best_dev_epoch
            states['best_dev_perf'] = self.trainer.best_dev_perf
            states['best_dev_step'] = self.trainer.best_dev_step
            states['best_metric_indicator'] = self.trainer.best_metric_indicator
        if self.recover_fitlog:
            try:
                import fitlog
                if fitlog._logger._log_dir is not None:
                    states['fitlog_log_dir'] = fitlog._logger._log_dir
                if fitlog._logger._save_log_dir is not None:
                    states['fitlog_save_log_dir'] = fitlog._logger._save_log_dir
            except:
                pass
        torch.save(states, self.save_path)
        logger.debug("Checkpoint:{} has been saved in epoch:{}.".format(self.save_path, self.epoch))

    def on_train_end(self):
        # 训练结束，根据情况删除保存的内容
        if self.delete_when_train_finish:
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
                logger.debug("Checkpoint:{} has been removed.".format(self.save_path))


class WarmupCallback(Callback):
    r"""
    learning rate按照一定的速率从0上升到设置的learning rate。
    """
    def __init__(self, warmup=0.1, schedule='constant'):
        r"""
        
        :param int,float warmup: 如果warmup为int，则在该step之前，learning rate根据schedule的策略变化; 如果warmup为float，
            如0.1, 则前10%的step是按照schedule策略调整learning rate。
        :param str schedule: 以哪种方式调整。
            linear: 前warmup的step上升到指定的learning rate(从Trainer中的optimizer处获取的), 后warmup的step下降到0；
            constant前warmup的step上升到指定learning rate，后面的step保持learning rate.
        """
        super().__init__()
        self.warmup = max(warmup, 0.)

        self.initial_lrs = []  # 存放param_group的learning rate
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")

    def _get_constant_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return 1

    def _get_linear_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)

    def on_train_begin(self):
        self.t_steps = (len(self.trainer.train_data) // (self.batch_size*self.update_every) +
                            int(len(self.trainer.train_data) % (self.batch_size*self.update_every)!= 0)) * self.n_epochs
        if self.warmup>1:
            self.warmup = self.warmup/self.t_steps
        self.t_steps = max(2, self.t_steps)  # 不能小于2
        # 获取param_group的初始learning rate
        for group in self.optimizer.param_groups:
            self.initial_lrs.append(group['lr'])

    def on_backward_end(self):
        if self.step%self.update_every==0:
            progress = (self.step/self.update_every)/self.t_steps
            for lr, group in zip(self.initial_lrs, self.optimizer.param_groups):
                group['lr'] = lr * self.get_lr(progress)


class SaveModelCallback(Callback):
    r"""
    由于Trainer在训练过程中只会保存最佳的模型， 该callback可实现多种方式的结果存储。
    会根据训练开始的时间戳在save_dir下建立文件夹，再在文件夹下存放多个模型::
        
        -save_dir
            -2019-07-03-15-06-36
                -epoch:0_step:20_{metric_key}:{evaluate_performance}.pt   # metric是给定的metric_key, evaluate_performance是性能
                -epoch:1_step:40_{metric_key}:{evaluate_performance}.pt
            -2019-07-03-15-10-00
                -epoch:0_step:20_{metric_key}:{evaluate_performance}.pt   # metric是给定的metric_key, evaluate_perfomance是性能
    """
    def __init__(self, save_dir, top=3, only_param=False, save_on_exception=False):
        r"""
        
        :param str save_dir: 将模型存放在哪个目录下，会在该目录下创建以时间戳命名的目录，并存放模型。如果save_dir不存在将自动创建
        :param int top: 保存dev表现top多少模型。-1为保存所有模型。
        :param bool only_param: 是否只保存模型的权重。
        :param save_on_exception: 发生exception时，是否保存一份发生exception的模型。模型名称为epoch:x_step:x_Exception:{exception_name}.
        """
        super().__init__()

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        if top < 0:
            self.top = sys.maxsize
        else:
            self.top = top
        self._ordered_save_models = []  # List[Tuple], Tuple[0]是metric， Tuple[1]是path。metric是依次变好的，所以从头删

        self.only_param = only_param
        self.save_on_exception = save_on_exception

    def on_train_begin(self):
        self.save_dir = os.path.join(self.save_dir, self.trainer.start_time)

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        metric_value = list(eval_result.values())[0][metric_key]
        self._save_this_model(metric_value)

    def _insert_into_ordered_save_models(self, pair):
        # pair:(metric_value, model_name)
        # 返回save的模型pair与删除的模型pair. pair中第一个元素是metric的值，第二个元素是模型的名称
        index = -1
        for _pair in self._ordered_save_models:
            if _pair[0]>=pair[0] and self.trainer.increase_better:
                break
            if not self.trainer.increase_better and _pair[0]<=pair[0]:
                break
            index += 1
        save_pair = None
        if len(self._ordered_save_models)<self.top or (len(self._ordered_save_models)>=self.top and index!=-1):
            save_pair = pair
            self._ordered_save_models.insert(index+1, pair)
        delete_pair = None
        if len(self._ordered_save_models)>self.top:
            delete_pair = self._ordered_save_models.pop(0)
        return save_pair, delete_pair

    def _save_this_model(self, metric_value):
        name = "epoch-{}_step-{}_{}-{:.6f}.pt".format(self.epoch, self.step, self.trainer.metric_key, metric_value)
        save_pair, delete_pair = self._insert_into_ordered_save_models((metric_value, name))
        if save_pair:
            try:
                _save_model(self.model, model_name=name, save_dir=self.save_dir, only_param=self.only_param)
            except Exception as e:
                logger.error(f"The following exception:{e} happens when save model to {self.save_dir}.")
        if delete_pair:
            try:
                delete_model_path = os.path.join(self.save_dir, delete_pair[1])
                if os.path.exists(delete_model_path):
                    os.remove(delete_model_path)
            except Exception as e:
                logger.error(f"Fail to delete model {name} at {self.save_dir} caused by exception:{e}.")

    def on_exception(self, exception):
        if self.save_on_exception:
            name = "epoch-{}_step-{}_Exception-{}.pt".format(self.epoch, self.step, exception.__class__.__name__)
            _save_model(self.model, model_name=name, save_dir=self.save_dir, only_param=self.only_param)


class CallbackException(BaseException):
    r"""
   当需要通过callback跳出训练的时候可以通过抛出CallbackException并在on_exception中捕获这个值。
   """
    
    def __init__(self, msg):
        r"""
        
        :param str msg: Exception的信息。
        """
        super(CallbackException, self).__init__(msg)


class EarlyStopError(CallbackException):
    r"""
    用于EarlyStop时从Trainer训练循环中跳出。
    
    """
    
    def __init__(self, msg):
        super(EarlyStopError, self).__init__(msg)


class EchoCallback(Callback):
    r"""
    用于测试分布式训练
    
    """
    def __init__(self, name, out=sys.stdout):
        super(EchoCallback, self).__init__()
        self.name = name
        self.out = out  # deprecated

    def __getattribute__(self, item):
        if item.startswith('on_'):
            logger.info('{}.{} has been called at pid: {}'.format(self.name, item, os.getpid()))
        return super(EchoCallback, self).__getattribute__(item)


class _TesterCallback(Callback):
    def __init__(self, data, model, metrics, metric_key=None, batch_size=16, num_workers=None):
        super(_TesterCallback, self).__init__()
        self.tester = Tester(data, model,
                             metrics=metrics, batch_size=batch_size,
                             num_workers=num_workers, verbose=0)
        if metric_key is not None:
            self.metric_key, self.increase_better = self._parse_metric_key(metric_key)
        else:
            self.metric_key = None
            self.increase_better = True
        self.score = None

    def on_valid_begin(self):
        cur_score = self.tester.test()
        eval_str = "Evaluation at Epoch {}/{}. Step:{}/{}. - {}".format(
                    self.epoch, self.n_epochs, self.step, self.n_steps,
                    self.tester._format_eval_results(cur_score))
        self.logger.info(eval_str)
        is_better = self.compare_better(cur_score)
        if is_better:
            self.score = cur_score
        return cur_score, is_better

    @staticmethod
    def _get_score(metric_dict, key):
        for metric in metric_dict.values():
            if key in metric:
                return metric[key]
        return None

    @staticmethod
    def _parse_metric_key(metric_key):
        # parse metric_key
        # increase_better is True. It means the exp result gets better if the indicator increases.
        # It is true by default.
        increase_better = False if metric_key[0] == "-" else True
        metric_key = metric_key[1:] if metric_key[0] == "+" or metric_key[0] == "-" else metric_key
        return metric_key, increase_better

    def compare_better(self, a):
        if self.score is None:
            return True
        if self.metric_key is None:
            metric_key = list(list(self.score.values())[0].keys())[0]
            self.metric_key, self.increase_better = self._parse_metric_key(metric_key)
        k = self.metric_key
        score = self._get_score(self.score, k)
        new_score = self._get_score(a, k)
        if score is None or new_score is None:
            return False
        if self.increase_better:
            return score <= new_score
        else:
            return score >= new_score
