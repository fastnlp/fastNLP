
__all__ = [
    'Callback',
]

from typing import Union, Callable, Dict, Optional, Any

from .callback_events import Events, EventsList, Filter
from fastNLP.core.callbacks.callback_events import _SingleEventState


class Callback:
    r"""
    实际使用的 callback 类，不管是我们 fastNLP 默认提供的一些 callback 类，还是用户自己定制的 callback 类，都应该继承该基类；
    callback 调用时机顺序大概如下
    Trainer.__init__():
        on_after_trainer_initialized()
    Trainer.run():
        if num_eval_sanity_batch>0:
            on_sanity_check_begin()  # 如果设置了num_eval_sanity_batch
            on_sanity_check_end()
        try:
            on_train_begin()
            while cur_epoch_idx < n_epochs:
                on_train_epoch_begin()
                while batch_idx_in_epoch<=num_batches_per_epoch:
                    on_fetch_data_begin()
                    on_fetch_data_end()
                    on_train_batch_begin()
                    on_before_backward()
                    on_after_backward()
                    on_before_zero_grad()  # 实际调用受到 accumulation_steps 影响
                    on_after_zero_grad()  # 实际调用受到 accumulation_steps 影响
                    on_before_optimizers_step()  # 实际调用受到 accumulation_steps 影响
                    on_after_optimizers_step()  # 实际调用受到 accumulation_steps 影响
                    on_train_batch_end()
                on_train_epoch_end()
        except BaseException:
            self.on_exception()
        finally:
            on_train_end()
    其它 callback 例如 on_evaluate_begin()/on_evaluate_end()将
    """

    def on_after_trainer_initialized(self, trainer, driver):
        r"""
        在 `Trainer` 初始化后会被触发；
        """
        pass

    def on_sanity_check_begin(self, trainer):
        r"""
        在 '预跑'检测 开始前会被触发；
        """
        pass

    def on_sanity_check_end(self, trainer, sanity_check_res):
        r"""
        在 '预跑'检测 开始后会被触发；

        :param trainer:
        :param sanity_check_res: 预跑的 evaluate 结果
        :return:
        """
        pass

    def on_train_begin(self, trainer):
        r"""
        在训练开始前会被触发；

        :param trainer:
        :return:
        """
        pass

    def on_train_end(self, trainer):
        r"""
        在训练完成后会被触发；

        :param trainer:
        :return:
        """
        pass

    def on_train_epoch_begin(self, trainer):
        r"""
        在训练过程中的每一个 epoch 开始前会被触发；

        :param trainer:
        :return:
        """
        pass

    def on_train_epoch_end(self, trainer):
        r"""
        在训练过程中的每一个 epoch 完成后会被触发；此时 trainer.cur_epoch_idx 已经完成加 1 操作。

        :param trainer:
        :return:
        """
        pass

    def on_fetch_data_begin(self, trainer):
        r"""
        在训练过程中准备取出下一个 batch 的数据时触发

        :param trainer:
        :return:
        """
        pass

    def on_fetch_data_end(self, trainer):
        r"""
        在训练过程中拿到当前的 batch 数据后会被触发；

        :param trainer:
        :return:
        """
        pass

    def on_train_batch_begin(self, trainer, batch, indices):
        r"""
        在取得数据，执行完 input_mapping (如果 Trainer 传有该参数），并且移动 batch 中的 tensor 到了指定设备。
            其中 batch 中的数据格式要么是 Dataloader 返回的每个 batch 的格式；要么是 input_mapping 之后的内容。
            如果 batch 是 dict 类型，直接增删其中的 key 或 修改其中的 value 会影响到输入到 model 的中的 batch 数据。

        :param trainer: `fastNLP.Trainer`
        :param batch: batch 的数据，已经经过 input_mapping (如果有) 以及 移动到指定设备 。
        :param list[int] indices: 当前的 batch 是 dataset 中的哪些数据
        """
        pass

    def on_train_batch_end(self, trainer):
        """
        完成一个 batch 的训练（forward）、梯度回传（backward）、梯度更新（step）、梯度置零、batch_idx_in_epoch与
            global_forward_batches累计加1操作。其中梯度更新】梯度置零操作会考虑 accumulation_steps ，所以不一定在当前 batch 会
            执行。

        :param trainer:
        :return:
        """
        pass

    def on_exception(self, trainer, exception):
        """
        在训练过程遇到异常时调用。

        :param trainer:
        :param exception: 遭遇的异常。
        :return:
        """
        pass

    def on_save_model(self, trainer):
        """
        当将要保存模型时调用，此刻模型还未保存。

        :param trainer:
        :return:
        """
        pass

    def on_load_model(self, trainer):
        """
        当将要加载模型时调用，此刻模型还未加载。

        :param trainer:
        :return:
        """
        pass

    def on_save_checkpoint(self, trainer) -> Dict:
        """
        当 Trainer 将要保存 checkpoint 的时候触发，该函数用于保存当前 callback 在恢复需要的相关数据。

        :param trainer:
        :return:
        """
        pass

    def on_load_checkpoint(self, trainer, states: Optional[Dict]):
        r"""
        当 Trainer 要恢复 checkpoint 的时候触发（ Trainer 与 Driver 已经加载好自身的状态），参数 states 为 on_save_checkpoint()
            的返回值。

        :param trainer:
        :param states:
        :return:
        """
        pass

    def on_before_backward(self, trainer, outputs):
        """
        在 backward 前执行。

        :param trainer:
        :param outputs: model 的返回内容。如果有 output_mapping ，则 outputs 中的内容为已经执行了 output_mapping 后的结果。
        :return:
        """
        pass

    def on_after_backward(self, trainer):
        """
        在 backward 后执行。在多卡场景下，由于 accumulation_steps 的影响，仅在需要真正 update 参数那次梯度回传才会触发梯度同步，
            因此在多卡且使用 accumulation_steps 时，可能存在某些 step 各卡上梯度不一致的问题。

        :param trainer:
        :return:
        """
        pass

    def on_before_optimizers_step(self, trainer, optimizers):
        """
        在进行 optimizer 优化进行前调用。该接口不一定每次前向计算都会触发，实际调用会受到 accumulation_steps 的影响。

        :param trainer:
        :param optimizers: 优化器，内容为在 Trainer 初始化时传入的值。
        :return:
        """
        pass

    def on_after_optimizers_step(self, trainer, optimizers):
        """
        在进行 optimizer 优化进行后调用。该接口不一定每次前向计算都会触发，实际调用会受到 accumulation_steps 的影响。

        :param trainer:
        :param optimizers: 优化器，内容为在 Trainer 初始化时传入的值。
        :return:
        """
        pass

    def on_before_zero_grad(self, trainer, optimizers):
        """
        在进行模型梯度置零前调用。该接口不一定每次前向计算都会触发，实际调用会受到 accumulation_steps 的影响。

        :param trainer:
        :param optimizers: 优化器，内容为在 Trainer 初始化时传入的值。
        :return:
        """
        pass

    def on_after_zero_grad(self, trainer, optimizers):
        """
        在进行模型梯度置零后调用。该接口不一定每次前向计算都会触发，实际调用会受到 accumulation_steps 的影响。

        :param trainer:
        :param optimizers: 优化器，内容为在 Trainer 初始化时传入的值。
        :return:
        """
        pass

    def on_evaluate_begin(self, trainer):
        """
        在将要进行 evaluate 时调用。如果是设置的以 step 数量 或 自定义地 决定 evaluate 的频率，该接口是在 on_train_batch_end 之后
            进行调用。如果是以 epoch 数量决定调用，该接口是在 on_train_epoch_end 之后调用。

        :param trainer:
        :return:
        """
        pass

    def on_evaluate_end(self, trainer, results):
        """
        结束 evaluate 时调用，并把 evaluate 的结果传入。

        :param trainer:
        :param results: Evaluate 的结果，一般是个 dict 。
        :return:
        """
        pass

    @property
    def callback_name(self):
        """
        callback 的名称，我们会使用该名称从 checkpoint 中读取的相应的 state 并传递给 on_load_checkpoint() 函数。

        :return:
        """
        return self.__class__.__name__

    @property
    def need_reproducible_sampler(self) -> bool:
        """
        当前 callback 是否需要能够复现的 sampler 。一般用于 checkpoint 类的 callback 。

        :return:
        """
        return False


class _CallbackWrapper(Callback):
    """
    对于用户使用函数修饰器加入的 callback 函数，使用该 _CallbackWrapper 类为其进行定制，这一个类只保留用户的
     这一个 callback 函数；
    """
    def __init__(self, event: Union[Events, EventsList], fn: Callable):
        r"""
        :param event: 具体的 callback 时机，例如 'on_train_begin' 等；可以多个时机，此时 `event` 的 type 应当为 'EventsList'；
        :param fn: 用户定制的 callback 函数；
        """

        self.fn = fn
        if isinstance(event, EventsList):
            for each_event in event:
                _filter = Filter(each_event.every, each_event.once, each_event.filter_fn)
                setattr(self, each_event.value, _filter(fn))
        elif isinstance(event, _SingleEventState):
            _filter = Filter(event.every, event.once, event.filter_fn)
            setattr(self, event.value, _filter(fn))

    @property
    def callback_name(self):
        return self.fn.__name__
