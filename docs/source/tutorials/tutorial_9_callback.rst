===================================================
使用Callback自定义你的训练过程
===================================================

在训练时，我们常常要使用trick来提高模型的性能（如调节学习率），或者要打印训练中的信息。
这里我们提供Callback类，在Trainer中插入代码，完成一些自定义的操作。

我们使用和 :doc:`/user/quickstart` 中一样的任务来进行详细的介绍。
给出一段评价性文字，预测其情感倾向是积极（label=1）、消极（label=0）还是中性（label=2），使用 :class:`~fastNLP.Trainer`  和  :class:`~fastNLP.Tester`  来进行快速训练和测试。
关于数据处理，Loss和Optimizer的选择可以看其他教程，这里仅在训练时加入学习率衰减。

---------------------
Callback的构建和使用
---------------------

创建Callback
    我们可以继承fastNLP :class:`~fastNLP.Callback` 类来定义自己的Callback。
    这里我们实现一个让学习率线性衰减的Callback。

    .. code-block:: python

        import fastNLP

        class LRDecay(fastNLP.Callback):
            def __init__(self):
                super(MyCallback, self).__init__()
                self.base_lrs = []
                self.delta = []

            def on_train_begin(self):
                # 初始化，仅训练开始时调用
                self.base_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
                self.delta = [float(lr) / self.n_epochs for lr in self.base_lrs]

            def on_epoch_end(self):
                # 每个epoch结束时，更新学习率
                ep = self.epoch
                lrs = [lr - d * ep for lr, d in zip(self.base_lrs, self.delta)]
                self.change_lr(lrs)

            def change_lr(self, lrs):
                for pg, lr in zip(self.optimizer.param_groups, lrs):
                    pg['lr'] = lr

    这里，:class:`~fastNLP.Callback` 中所有以 ``on_`` 开头的类方法会在 :class:`~fastNLP.Trainer` 的训练中在特定时间调用。
    如 on_train_begin() 会在训练开始时被调用，on_epoch_end() 会在每个 epoch 结束时调用。
    具体有哪些类方法，参见文档 :class:`~fastNLP.Callback` 。

    另外，为了使用方便，可以在 :class:`~fastNLP.Callback` 内部访问 :class:`~fastNLP.Trainer` 中的属性，如 optimizer, epoch, step，分别对应训练时的优化器，当前epoch数，和当前的总step数。
    具体可访问的属性，参见文档 :class:`~fastNLP.Callback` 。

使用Callback
    在定义好 :class:`~fastNLP.Callback` 之后，就能将它传入Trainer的 ``callbacks`` 参数，在实际训练时使用。

    .. code-block:: python

        """
        数据预处理，模型定义等等
        """

        trainer = fastNLP.Trainer(
            model=model, train_data=train_data, dev_data=dev_data,
            optimizer=optimizer, metrics=metrics,
            batch_size=10, n_epochs=100,
            callbacks=[LRDecay()])

        trainer.train()
