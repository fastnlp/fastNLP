__all__ = ['Loop']


class Loop:
    r"""
    :class:`.TrainBatchLoop` 和 :class:`.EvaluateBatchLoop` 的父类，您可以继承此
    类来定制自己的训练或者评测 ``loop``。

    为了在实现 **fastNLP** 主要功能的同时保证 **fastNLP** 的易用性和代码的易读性，
    我们只对训练中的循环做了非常简单的抽象，``Loop`` 表示的是在训练或者评测的过程中针
    对单独一个 ``dataloader`` 的一个 ``epoch`` 的运算过程；

    更为具体的使用详见 :class:`.TrainBatchLoop` 和 :class:`.EvaluateBatchLoop`。
    """

    def run(self, controller, dataloader):
        r"""
        遍历参数 ``dataloader`` 的所有数据，使用 ``controller`` 进行训练或者评测；

        .. note::

            :class:`.Trainer` 和 :class:`.Evaluator` 中都提供了方便您进行定制
            ``Loop`` 的接口函数，例如 :meth:`.Trainer.train_step`，:meth:`.\
            Trainer.backward` 等；

            在定制您自己的 ``TrainBatchLoop`` 时，请务必记得在正确的时机调用对应的
            callback 函数，详见 :class:`.TrainBatchLoop` 中对于 callback 函数的调
            用；

        """

    @staticmethod
    def batch_step_fn(controller, batch):
        r"""
        对于具体的一个 ``batch`` 的数据，实现训练或者评测过程中的一步；
        """
