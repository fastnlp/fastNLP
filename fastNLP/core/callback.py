class Callback(object):
    """An Interface for all callbacks.

    Any customized callback should implement at least one of the following methods.

    """

    def __init__(self):
        super(Callback, self).__init__()

    def before_train(self, *args):
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
        # after gradient backward, before optimizer step
        pass

    def after_step(self):
        # after optimizer step
        pass

    def after_batch(self):
        # at the end of each step/mini-batch
        pass

    def after_valid(self, eval_result, metric_key, optimizer):
        # after validation
        pass

    def after_epoch(self, cur_epoch, n_epoch, optimizer):
        # at the end of each epoch
        pass

    def after_train(self, model):
        # after training loop
        pass

    def on_exception(self, exception, model, indices):
        # when exception raised in training
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
    def after_step(self):
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
        print("cur_epoch: ", cur_epoch)
        print("total_epoch: ", total_epoch)

    def before_batch(self, batch_x, batch_y, indices):
        print("before_batch")
        print("batch_x:", batch_x)
        print("batch_y:", batch_y)
        print("indices: ", indices)

    def before_loss(self, batch_y, predict_y):
        print("before_loss")
        print("batch_y: ", batch_y)
        print("predict_y: ", predict_y)

    def before_backward(self, loss, model):
        print("before_backward")
        print("loss=", loss)
        print("model: ", model)

    def after_batch(self):
        print("after_batch")

    def after_epoch(self, cur_epoch, n_epoch, optimizer):
        print("after_epoch")
        print("cur_epoch: ", cur_epoch)
        print("n_epoch: ", n_epoch)
        print("optimizer", optimizer)

    def after_train(self, model):
        print("after_train")
        print("model: ", model)


if __name__ == "__main__":
    manager = CallbackManager(env={"n_epoch": 3}, callbacks=[DummyCallback(), DummyCallback()])
    manager.after_epoch(3, 10, "optimizer")
