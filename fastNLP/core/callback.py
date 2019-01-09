class Callback(object):
    """An Interface for all callbacks.

    Any customized callback should implement at least one of the following methods.

    """

    def __init__(self):
        super(Callback, self).__init__()

    def before_train(self):
        # before the main training loop
        pass

    def before_epoch(self):
        # at the beginning of each epoch
        pass

    def before_batch(self):
        # at the beginning of each step/mini-batch
        pass

    def before_loss(self):
        # after data_forward, and before loss computation
        pass

    def before_backward(self):
        # after loss computation, and before gradient backward
        pass

    def after_batch(self):
        # at the end of each step/mini-batch
        pass

    def after_epoch(self):
        # at the end of each epoch
        pass

    def after_train(self):
        # after training loop
        pass


def transfer(func):
    """装饰器，将对CallbackManager的调用转发到各个Callback子类.

    :param func:
    :return:
    """

    def wrapper(manager):
        returns = []
        for callback in manager.callbacks:
            for env_name, env_value in manager.env.items():
                setattr(callback, env_name, env_value)
            returns.append(getattr(callback, func.__name__)())
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
    def before_epoch(self):
        pass

    @transfer
    def before_batch(self):
        pass

    @transfer
    def before_loss(self):
        pass

    @transfer
    def before_backward(self):
        pass

    @transfer
    def after_batch(self):
        pass

    @transfer
    def after_epoch(self):
        pass

    @transfer
    def after_train(self):
        pass


class DummyCallback(Callback):
    def before_train(self):
        print("before train!!!")
        print(self.n_epoch)

    def after_epoch(self):
        print("after epoch!!!")
        return 12


class EchoCallback(Callback):
    def before_train(self):
        print("before_train")

    def before_epoch(self):
        print("before_epoch")

    def before_batch(self):
        print("before_batch")

    def before_loss(self):
        print("before_loss")

    def before_backward(self):
        print("before_backward")

    def after_batch(self):
        print("after_batch")

    def after_epoch(self):
        print("after_epoch")

    def after_train(self):
        print("after_train")


if __name__ == "__main__":
    manager = CallbackManager(env={"n_epoch": 3}, callbacks=[DummyCallback(), DummyCallback()])
    manager.before_train()
    print(manager.after_epoch())
