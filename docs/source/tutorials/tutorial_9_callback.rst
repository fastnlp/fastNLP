
==============================================================================
Callback 教程
==============================================================================

在训练时，我们常常要使用trick来提高模型的性能（如调节学习率），或者要打印训练中的信息。
这里我们提供Callback类，在Trainer中插入代码，完成一些自定义的操作。
我们使用和 :doc:`/user/quickstart` 中一样的任务来进行详细的介绍。
给出一段评价性文字，预测其情感倾向是积极（label=1）、消极（label=0）还是中性（label=2），使用 :class:`~fastNLP.Trainer`  和  :class:`~fastNLP.Tester`  来进行快速训练和测试。
关于数据处理，Loss和Optimizer的选择可以看其他教程，这里仅在训练时加入学习率衰减。

---------------------
Callback的构建
---------------------

创建Callback
    我们可以继承fastNLP :class:`~fastNLP.Callback` 类来定义自己的Callback。
    这里我们先实现一个让学习率线性衰减的Callback。

    .. code-block:: python

        import fastNLP

        class MyCallback(fastNLP.Callback):
            def 
