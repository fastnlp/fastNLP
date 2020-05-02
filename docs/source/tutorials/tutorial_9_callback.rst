===================================================
使用 Callback 自定义你的训练过程
===================================================

- `什么是Callback`_
- `使用 Callback`_
- `fastNLP 中的 Callback`_
- `自定义 Callback`_


什么是Callback
---------------------

:class:`~fastNLP.core.callback.Callback` 是与 :class:`~fastNLP.core.trainer.Trainer` 紧密结合的模块，利用 Callback 可以在 :class:`~fastNLP.core.trainer.Trainer` 训练时，加入自定义的操作，比如梯度裁剪，学习率调节，测试模型的性能等。定义的 Callback 会在训练的特定阶段被调用。

fastNLP 中提供了很多常用的 :class:`~fastNLP.core.callback.Callback` ，开箱即用。


使用 Callback
---------------------

使用 Callback 很简单，将需要的 callback 按 list 存储，以对应参数 ``callbacks`` 传入对应的 Trainer。Trainer 在训练时就会自动执行这些 Callback 指定的操作了。


.. code-block:: python

    from fastNLP import (Callback, EarlyStopCallback,
                         Trainer, CrossEntropyLoss, AccuracyMetric)
    from fastNLP.models import CNNText
    import torch.cuda

    # prepare data
    def get_data():
        from fastNLP.io import ChnSentiCorpPipe as pipe
        data = pipe().process_from_file()
        print(data)
        data.rename_field('chars', 'words')
        train_data = data.get_dataset('train')
        dev_data = data.get_dataset('dev')
        test_data = data.get_dataset('test')
        vocab = data.get_vocab('words')
        tgt_vocab = data.get_vocab('target')
        return train_data, dev_data, test_data, vocab, tgt_vocab

    # prepare model
    train_data, dev_data, _, vocab, tgt_vocab = get_data()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = CNNText((len(vocab),50), num_classes=len(tgt_vocab))

    # define callback
    callbacks=[EarlyStopCallback(5)]

    # pass callbacks to Trainer
    def train_with_callback(cb_list):
        trainer = Trainer(
            device=device,
            n_epochs=3,
            model=model,
            train_data=train_data,
            dev_data=dev_data,
            loss=CrossEntropyLoss(),
            metrics=AccuracyMetric(),
            callbacks=cb_list,
            check_code_level=-1
        )
        trainer.train()

    train_with_callback(callbacks)



fastNLP 中的 Callback
---------------------

fastNLP 中提供了很多常用的 Callback，如梯度裁剪，训练时早停和测试验证集，fitlog 等等。具体 Callback 请参考 :mod:`fastNLP.core.callback`

.. code-block:: python

    from fastNLP import EarlyStopCallback, GradientClipCallback, EvaluateCallback
    callbacks = [
        EarlyStopCallback(5),
        GradientClipCallback(clip_value=5, clip_type='value'),
        EvaluateCallback(dev_data)
    ]

    train_with_callback(callbacks)

自定义 Callback
---------------------

这里我们以一个简单的 Callback作为例子，它的作用是打印每一个 Epoch 平均训练 loss。

1. 创建 Callback
    
    要自定义 Callback，我们要实现一个类，继承 :class:`~fastNLP.core.callback.Callback` 。这里我们定义 ``MyCallBack`` ，继承 fastNLP.Callback 。

2. 指定 Callback 调用的阶段
    
    Callback 中所有以 `on_` 开头的类方法会在 Trainer 的训练中在特定阶段调用。 如 on_train_begin() 会在训练开始时被调用，on_epoch_end()
    会在每个 epoch 结束时调用。 具体有哪些类方法，参见 :class:`~fastNLP.core.callback.Callback` 文档。这里， MyCallBack 在求得loss时调用 on_backward_begin() 记录
    当前 loss，在每一个 epoch 结束时调用 on_epoch_end() ，求当前 epoch 平均loss并输出。

3. 使用 Callback 的属性访问 Trainer 的内部信息
    
    为了方便使用，可以使用 :class:`~fastNLP.core.callback.Callback` 的属性，访问 :class:`~fastNLP.core.trainer.Trainer` 中的对应信息，如 optimizer, epoch, n_epochs，分别对应训练时的优化器，
    当前 epoch 数，和总 epoch 数。 具体可访问的属性，参见 :class:`~fastNLP.core.callback.Callback` 。这里， MyCallBack 为了求平均 loss ，需要知道当前 epoch 的总步
    数，可以通过 self.step 属性得到当前训练了多少步。

.. code-block:: python

    from fastNLP import Callback
    from fastNLP import logger

    class MyCallBack(Callback):
        """Print average loss in each epoch"""
        def __init__(self):
            super().__init__()
            self.total_loss = 0
            self.start_step = 0

        def on_backward_begin(self, loss):
            self.total_loss += loss.item()

        def on_epoch_end(self):
            n_steps = self.step - self.start_step
            avg_loss = self.total_loss / n_steps
            logger.info('Avg loss at epoch %d, %.6f', self.epoch, avg_loss)
            self.start_step = self.step

    callbacks = [MyCallBack()]
    train_with_callback(callbacks)


----------------------------------
代码下载
----------------------------------

.. raw:: html

    <a href="../_static/notebooks/tutorial_9_callback.ipynb" download="tutorial_9_callback.ipynb">点击下载 IPython Notebook 文件</a><hr>
