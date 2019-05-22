r"""
callback模块实现了 fastNLP 中的许多 callback 类，用于增强 :class:`~fastNLP.Trainer` 类。

虽然Trainer本身已经集成了一些功能，但仍然不足以囊括训练过程中可能需要到的功能，
比如负采样，learning rate decay, Early Stop等。
为了解决这个问题fastNLP引入了callback的机制，Callback 是一种在Trainer训练过程中特定阶段会运行的函数集合。
关于Trainer的详细文档，请参见 :doc:`trainer 模块<fastNLP.core.trainer>`

我们将 :meth:`~fastNLP.Train.train` 这个函数内部分为以下的阶段，在对应阶段会触发相应的调用::

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

如下面的例子所示，我们可以使用内置的 callback 类，或者继承 :class:`~fastNLP.core.callback.Callback`
定义自己的 callback 类::
    
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
    "TensorboardCallback",
    "LRScheduler",
    "ControlC",
    
    "CallbackException",
    "EarlyStopError"
]

import os

import torch
from copy import deepcopy
try:
    from tensorboardX import SummaryWriter
    
    tensorboardX_flag = True
except:
    tensorboardX_flag = False

from ..io.model_io import ModelSaver, ModelLoader
from .dataset import DataSet
from .tester import Tester

try:
    import fitlog
except:
    pass

class Callback(object):
    """
    别名：:class:`fastNLP.Callback` :class:`fastNLP.core.callback.Callback`

    Callback是fastNLP中被设计用于增强 :class:`~fastNLP.Trainer` 的类。
    如果Callback被传递给了 Trainer , 则 Trainer 会在对应的阶段调用Callback的函数，
    具体调用时机可以通过 :doc:`trainer 模块<fastNLP.core.trainer>` 查看。
    这是Callback的基类，所有的callback必须继承自这个类

    """
    
    def __init__(self):
        super(Callback, self).__init__()
        self._trainer = None  # 在Trainer内部被重新赋值
    
    @property
    def trainer(self):
        """
        该属性可以通过self.trainer获取到，一般情况下不需要使用这个属性。
        """
        return self._trainer
    
    @property
    def step(self):
        """当前运行到的step, 范围为[1, self.n_steps+1)"""
        return self._trainer.step
    
    @property
    def n_steps(self):
        """Trainer一共会运行多少步"""
        return self._trainer.n_steps
    
    @property
    def batch_size(self):
        """train和evaluate时的batch_size为多大"""
        return self._trainer.batch_size
    
    @property
    def epoch(self):
        """当前运行的epoch数，范围是[1, self.n_epochs+1)"""
        return self._trainer.epoch
    
    @property
    def n_epochs(self):
        """一共会运行多少个epoch"""
        return self._trainer.n_epochs
    
    @property
    def optimizer(self):
        """初始化Trainer时传递的Optimizer"""
        return self._trainer.optimizer
    
    @property
    def model(self):
        """正在被Trainer训练的模型"""
        return self._trainer.model
    
    @property
    def pbar(self):
        """如果在Callback中需要打印内容，请使用self.pbar.write(str)。否则可能出现命令行显示效果不太好的问题。在
        on_train_begin(), on_train_end(), on_exception()中请不要使用该属性，通过print输出即可。"""
        return self._trainer.pbar
    
    @property
    def update_every(self):
        """Trainer中的模型多少次反向传播才进行一次梯度更新，在Trainer初始化时传入的。"""
        return self._trainer.update_every
    
    @property
    def batch_per_epoch(self):
        """每个epoch一共有多少个batch，只有在on_epoch_begin之后才能调用该属性。"""
        return self._trainer.batch_per_epoch
    
    def on_train_begin(self):
        """
        在Train过程开始之前调用。

        :return:
        """
        pass
    
    def on_epoch_begin(self):
        """
        在每个epoch开始之前调用一次

        :return:
        """
        pass
    
    def on_batch_begin(self, batch_x, batch_y, indices):
        """
        每次采集到一个batch的数据则调用一次。这里对batch_x或batch_y删除添加内容是可以影响到Trainer中内容的。所以在这一步
        可以进行一些负采样之类的操作

        :param dict batch_x: DataSet中被设置为input的field的batch。
        :param dict batch_y: DataSet中被设置为target的field的batch。
        :param list(int) indices: 这次采样使用到的indices，可以通过DataSet[indices]获取出这个batch采出的Instance，在一些
            情况下可以帮助定位是哪个Sample导致了错误。仅在Trainer的prefetch为False时可用。
        :return:
        """
        pass
    
    def on_loss_begin(self, batch_y, predict_y):
        """
        在计算loss前调用，即这里修改batch_y或predict_y的值是可以影响到loss计算的。

        :param dict batch_y: 在DataSet中被设置为target的field的batch集合。
        :param dict predict_y: 模型的forward()返回的结果。
        :return:
        """
        pass
    
    def on_backward_begin(self, loss):
        """
        在loss得到之后，但在反向传播之前。可能可以进行loss是否为NaN的检查。

        :param torch.Tensor loss: 计算得到的loss值
        :return:
        """
        pass
    
    def on_backward_end(self):
        """
        反向梯度传播已完成，但由于update_every的设置，可能并不是每一次调用都有梯度。到这一步，还没有更新参数。

        :return:
        """
        pass
    
    def on_step_end(self):
        """
        到这里模型的参数已经按照梯度更新。但可能受update_every影响，并不是每次都更新了。

        :return:
        """
        pass
    
    def on_batch_end(self):
        """
        这一步与on_step_end是紧接着的。只是为了对称性加上了这一步。

        """
        pass
    
    def on_valid_begin(self):
        """
        如果Trainer中设置了验证，则发生验证前会调用该函数

        :return:
        """
        pass
    
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        """
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
        """
        每个epoch结束将会调用该方法
        """
        pass
    
    def on_train_end(self):
        """
        训练结束，调用该方法
        """
        pass
    
    def on_exception(self, exception):
        """
        当训练过程出现异常，会触发该方法
        :param exception: 某种类型的Exception，比如KeyboardInterrupt等
        """
        pass


def _transfer(func):
    """装饰器，将对CallbackManager的调用转发到各个Callback子类.
    
    :param func:
    :return:
    """
    
    def wrapper(manager, *arg):
        returns = []
        for callback in manager.callbacks:
            returns.append(getattr(callback, func.__name__)(*arg))
        return returns
    
    return wrapper


class CallbackManager(Callback):
    def __init__(self, env, callbacks=None):
        """
        内部使用的Callback管理类

        :param dict env: The key is the name of the Trainer attribute(str). The value is the attribute itself.
        :param List[Callback] callbacks:
        """
        super(CallbackManager, self).__init__()
        # set attribute of trainer environment
        
        self.callbacks = []
        if callbacks is not None:
            if isinstance(callbacks, list):
                if all([isinstance(cb, Callback) for cb in callbacks]) is True:
                    self.callbacks.extend(callbacks)
                else:
                    obj = [not isinstance(cb, Callback) for cb in callbacks][0]
                    raise TypeError(f"Expect sub-classes of Callback. Got {type(obj)}")
            else:
                raise TypeError(f"Expect callbacks in CallbackManager(callbacks) to be list. Got {type(callbacks)}.")
        
        for env_name, env_val in env.items():
            for callback in self.callbacks:
                setattr(callback, '_' + env_name, env_val)  # Callback.trainer
    
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
    def on_epoch_end(self):
        pass
    
    @_transfer
    def on_train_end(self):
        pass
    
    @_transfer
    def on_exception(self, exception):
        pass


class GradientClipCallback(Callback):
    """
    别名：:class:`fastNLP.GradientClipCallback` :class:`fastNLP.core.callback.GradientClipCallback`

    每次backward前，将parameter的gradient clip到某个范围。

    :param None,torch.Tensor,List[torch.Tensor] parameters: 一般通过model.parameters()获得。如果为None则默认对Trainer
        的model中所有参数进行clip
    :param float clip_value: 将gradient 限制到[-clip_value, clip_value]。clip_value应该为正数
    :param str clip_type: 支持'norm', 'value'
        两种::

            1 'norm', 将gradient的norm rescale到[-clip_value, clip_value]
        
            2 'value', 将gradient限制在[-clip_value, clip_value], 小于-clip_value的gradient被赋值为-clip_value;
            大于clip_value的gradient被赋值为clip_value.

    """
    
    def __init__(self, parameters=None, clip_value=1, clip_type='norm'):
        
        super().__init__()
        
        from torch import nn
        if clip_type == 'norm':
            self.clip_fun = nn.utils.clip_grad_norm_
        elif clip_type == 'value':
            self.clip_fun = nn.utils.clip_grad_value_
        else:
            raise ValueError("Only supports `norm` or `value` right now.")
        self.parameters = parameters
        self.clip_value = clip_value
    
    def on_backward_end(self):
        if self.parameters is None:
            self.clip_fun(self.model.parameters(), self.clip_value)
        else:
            self.clip_fun(self.parameters, self.clip_value)


class EarlyStopCallback(Callback):
    """
    别名：:class:`fastNLP.EarlyStopCallback` :class:`fastNLP.core.callback.EarlyStopCallback`
    
    多少个epoch没有变好就停止训练，相关类 :class:`EarlyStopError`

    :param int patience: epoch的数量
    """
    
    def __init__(self, patience):
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
            print("Early Stopping triggered in epoch {}!".format(self.epoch))
        else:
            raise exception  # 抛出陌生Error

class FitlogCallback(Callback):
    """
    别名: :class:`fastNLP.FitlogCallback` :class:`fastNLP.core.callback.FitlogCallback`

    该callback将loss和progress自动写入到fitlog中; 如果Trainer有dev的数据，将自动把dev的结果写入到log中; 同时还支持传入
        一个(或多个)test数据集进行测试(只有在trainer具有dev时才能使用)，每次在dev上evaluate之后会在这些数据集上验证一下。
        并将验证结果写入到fitlog中。这些数据集的结果是根据dev上最好的结果报道的，即如果dev在第3个epoch取得了最佳，则
        fitlog中记录的关于这些数据集的结果就是来自第三个epoch的结果。

    :param DataSet,dict(DataSet) data: 传入DataSet对象，会使用多个Trainer中的metric对数据进行验证。如果需要传入多个
        DataSet请通过dict的方式传入，dict的key将作为对应dataset的name传递给fitlog。若tester不为None时，data需要通过
        dict的方式传入。如果仅传入DataSet, 则被命名为test
    :param Tester tester: Tester对象，将在on_valid_end时调用。tester中的DataSet会被称为为`test`
    :param int verbose: 是否在终端打印内容，0不打印
    :param bool log_exception: fitlog是否记录发生的exception信息
    """

    def __init__(self, data=None, tester=None, verbose=0, log_exception=False):
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self._log_exception = log_exception
        if tester is not None:
            assert isinstance(tester, Tester), "Only fastNLP.Tester allowed."
            assert isinstance(data, dict) or data is None, "If tester is not None, only dict[DataSet] allowed for data."
            if data is not None:
                assert 'test' not in data, "Cannot use `test` as DataSet key, when tester is passed."
            setattr(tester, 'verbose', 0)
            self.testers['test'] = tester

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets[key] = value
        elif isinstance(data, DataSet):
            self.datasets['test'] = data
        else:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

        self.verbose = verbose

    def on_train_begin(self):
        if (len(self.datasets)>0 or len(self.testers)>0 ) and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra data to do evaluation.")

        if len(self.datasets)>0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model, batch_size=self.batch_size, metrics=self.trainer.metrics,
                                verbose=0)
                self.testers[key] = tester
        fitlog.add_progress(total_steps=self.n_steps)

    def on_backward_begin(self, loss):
        fitlog.add_loss(loss.item(), name='loss', step=self.step, epoch=self.epoch)

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        if len(self.testers)>0:
            for key, tester in self.testers.items():
                try:
                    eval_result = tester.test()
                    if self.verbose!=0:
                        self.pbar.write("Evaluation on DataSet {}:".format(key))
                        self.pbar.write(tester._format_eval_results(eval_result))
                    fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                    if better_result:
                        fitlog.add_best_metric(eval_result, name=key)
                except Exception:
                    self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))

    def on_train_end(self):
        fitlog.finish()

    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(str(exception), name='except_info')


class LRScheduler(Callback):
    """
    别名：:class:`fastNLP.LRScheduler` :class:`fastNLP.core.callback.LRScheduler`

    对PyTorch LR Scheduler的包装以使得其可以被Trainer所使用

    :param torch.optim.lr_scheduler._LRScheduler lr_scheduler: PyTorch的lr_scheduler
    """
    
    def __init__(self, lr_scheduler):
        
        super(LRScheduler, self).__init__()
        import torch.optim
        if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            self.scheduler = lr_scheduler
        else:
            raise ValueError(f"Expect torch.optim.lr_scheduler for LRScheduler. Got {type(lr_scheduler)}.")
    
    def on_epoch_begin(self):
        self.scheduler.step(self.epoch)


class ControlC(Callback):
    """
    别名：:class:`fastNLP.ControlC` :class:`fastNLP.core.callback.ControlC`

    :param bool quit_all: 若为True,则检测到control+C 直接退出程序；否则只退出Trainer
    """
    
    def __init__(self, quit_all):
        
        super(ControlC, self).__init__()
        if type(quit_all) != bool:
            raise ValueError("In KeyBoardInterrupt, quit_all arguemnt must be a bool.")
        self.quit_all = quit_all
    
    def on_exception(self, exception):
        if isinstance(exception, KeyboardInterrupt):
            if self.quit_all is True:
                import sys
                sys.exit(0)  # 直接退出程序
            else:
                pass
        else:
            raise exception  # 抛出陌生Error


class SmoothValue(object):
    def __init__(self, beta: float):
        self.beta, self.n, self.mov_avg = beta, 0, 0
        self.smooth = None
    
    def add_value(self, val: float) -> None:
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


class LRFinder(Callback):
    """
    别名：:class:`fastNLP.LRFinder` :class:`fastNLP.core.callback.LRFinder`

    用第一个 epoch 找最佳的学习率，从第二个epoch开始应用它

    :param float start_lr: 学习率下界
    :param float end_lr: 学习率上界
    """
    
    def __init__(self, start_lr=1e-6, end_lr=10):
        
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        
        self.stop = False
        self.best_loss = 0.
        self.best_lr = None
        self.loss_history = []
        self.smooth_value = SmoothValue(0.8)
        self.opt = None
        self.find = None
        self.loader = ModelLoader()
    
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
            ModelSaver("tmp").save_pytorch(self.trainer.model, param_only=True)
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
            ModelLoader().load_pytorch(self.trainer.model, "tmp")
            self.pbar.write("Model reset. \nFind best lr={}".format(self.best_lr))


class TensorboardCallback(Callback):
    """
    别名：:class:`fastNLP.TensorboardCallback` :class:`fastNLP.core.callback.TensorboardCallback`

    接受以下一个或多个字符串作为参数：
    - "model"
    - "loss"
    - "metric"
    
    .. warning::
        fastNLP 已停止对此功能的维护，请等待 fastNLP 兼容 PyTorch1.1 的下一个版本。
        或者使用和 fastNLP 高度配合的 fitlog（参见 :doc:`/user/with_fitlog` ）。
        
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


class CallbackException(BaseException):
    """
   当需要通过callback跳出训练的时候可以通过抛出CallbackException并在on_exception中捕获这个值。

   :param str msg: Exception的信息。
   """
    
    def __init__(self, msg):
        super(CallbackException, self).__init__(msg)


class EarlyStopError(CallbackException):
    """
    用于EarlyStop时从Trainer训练循环中跳出。
    
    """
    
    def __init__(self, msg):
        super(EarlyStopError, self).__init__(msg)
