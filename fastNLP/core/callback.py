class Callback(object):
    """An Interface for all callbacks.

    Any customized callback should implement at least one of the following methods.

    """

    def __init__(self):
        super(Callback, self).__init__()

    def before_train(self):
        # before the main training loop
        pass

    def before_epoch(self, cur_epoch, total_epoch):
        # at the beginning of each epoch
        pass

    def before_batch(self, batch_x, batch_y, indices):
        # at the beginning of each step/mini-batch
        pass

    def before_loss(self, batch_y, predict_y):
        # after data_forward, and before loss computation
        pass

    def before_backward(self, loss, model):
        # after loss computation, and before gradient backward
        pass

    def after_backward(self, model):
        pass

    def after_step(self, optimizer):
        pass

    def after_batch(self, *args):
        # at the end of each step/mini-batch
        pass

    def after_valid(self, eval_result, metric_key, optimizer):
        """
        每次执行验证机的evaluation后会调用。传入eval_result

        :param eval_result: Dict[str: Dict[str: float]], evaluation的结果
        :param metric_key: str
        :param optimizer:
        :return:
        """
        pass

    def after_epoch(self, cur_epoch, n_epoch, optimizer):
        """
        每个epoch结束将会调用该方法

        :param cur_epoch: int, 当前的batch。从1开始。
        :param n_epoch: int, 总的batch数
        :param optimizer: 传入Trainer的optimizer。
        :return:
        """
        pass

    def after_train(self, model):
        """
        训练结束，调用该方法

        :param model: nn.Module, 传入Trainer的模型
        :return:
        """
        pass

    def on_exception(self, exception, model, indices):
        """
        当训练过程出现异常，会触发该方法
        :param exception: 某种类型的Exception，比如KeyboardInterrupt等
        :param model: 传入Trainer的模型
        :param indices: 当前batch的index
        :return:
        """
        pass

def transfer(func):
    """装饰器，将对CallbackManager的调用转发到各个Callback子类.

    :param func:
    :return:
    """

    def wrapper(manager, *arg):
        returns = []
        for callback in manager.callbacks:
            for env_name, env_value in manager.env.items():
                setattr(callback, env_name, env_value)
            returns.append(getattr(callback, func.__name__)(*arg))
        return returns

    return wrapper


class CallbackManager(Callback):
    """A manager for all callbacks passed into Trainer.
    It collects resources inside Trainer and raise callbacks.

    """

    def __init__(self, env, callbacks=None):
        """

        :param dict env: The key is the name of the Trainer attribute(str). The value is the attribute itself.
        :param Callback callbacks:
        """
        super(CallbackManager, self).__init__()
        # set attribute of trainer environment
        self.env = env

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

    @transfer
    def before_train(self):
        pass

    @transfer
    def before_epoch(self, cur_epoch, total_epoch):
        pass

    @transfer
    def before_batch(self, batch_x, batch_y, indices):
        pass

    @transfer
    def before_loss(self, batch_y, predict_y):
        pass

    @transfer
    def before_backward(self, loss, model):
        pass

    @transfer
    def after_backward(self, model):
        pass

    @transfer
    def after_step(self, optimizer):
        pass

    @transfer
    def after_batch(self):
        pass

    @transfer
    def after_valid(self, eval_result, metric_key, optimizer):
        pass

    @transfer
    def after_epoch(self, cur_epoch, n_epoch, optimizer):
        pass

    @transfer
    def after_train(self, model):
        pass

    @transfer
    def on_exception(self, exception, model, indices):
        pass


class DummyCallback(Callback):
    def before_train(self, *arg):
        print(arg)

    def after_epoch(self, cur_epoch, n_epoch, optimizer):
        print(cur_epoch, n_epoch, optimizer)


class EchoCallback(Callback):
    def before_train(self):
        print("before_train")

    def before_epoch(self, cur_epoch, total_epoch):
        print("before_epoch")

    def before_batch(self, batch_x, batch_y, indices):
        print("before_batch")

    def before_loss(self, batch_y, predict_y):
        print("before_loss")

    def before_backward(self, loss, model):
        print("before_backward")

    def after_batch(self):
        print("after_batch")

    def after_epoch(self, cur_epoch, n_epoch, optimizer):
        print("after_epoch")

    def after_train(self, model):
        print("after_train")

class GradientClipCallback(Callback):
    def __init__(self, parameters=None, clip_value=1, clip_type='norm'):
        """
        每次backward前，将parameter的gradient clip到某个范围。

        :param parameters: None, torch.Tensor或List[torch.Tensor], 一般通过model.parameters()获得。如果为None则默认对Trainer
            的model中所有参数进行clip
        :param clip_value: float, 将gradient 限制到[-clip_value, clip_value]。clip_value应该为正数
        :param clip_type: str, 支持'norm', 'value'两种。
            (1) 'norm', 将gradient的norm rescale到[-clip_value, clip_value]
            (2) 'value', 将gradient限制在[-clip_value, clip_value], 小于-clip_value的gradient被赋值为-clip_value; 大于
                clip_value的gradient被赋值为clip_value.
        """
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

    def after_backward(self, model):
        self.clip_fun(model.parameters(), self.clip_value)



if __name__ == "__main__":
    manager = CallbackManager(env={"n_epoch": 3}, callbacks=[DummyCallback(), DummyCallback()])
    manager.before_train(10, 11, 12)
    # print(manager.after_epoch())
