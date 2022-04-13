"""

该 Module 用来实现一个用于记载用户 callback 实时数据的 state，该 state 实际上是一个 字典，我们通过复用 __getattr__ 方法来实现类似
类属性的字典调用方式；

提供该类的主要目的在于与 Filter 中的特殊的 filter_fn 合作，方便用户能够使用到自己想要的一切特殊的定制方式；

这一特殊的 Filter 实现需要用户记录一些特殊的状态值，例如 accuracy 等，但是我们不希望用户将这些状态值直接挂在 trainer 实例上，因为这样会
污染 trainer 自己的类属性，从而可能导致一些莫名其妙的 bug；

我们开放 state 用于用户这一特殊的定制选择；
"""
from dataclasses import dataclass
from typing import Optional, Dict


__all__ = [
    'State',
    'TrainerState'
]


class State(dict):
    r"""
    提供给用户使用的 state；

    为了实现断点重训，用户应当保证其保存的信息都是可序列化的；

    推荐的使用方式：
        >>> state = State()
        >>> state["best_accuracy"] = 0.9
        >>> print(state["best_accuracy"])
        or
        >>> print(state.best_accuracy)
    """

    __slots__ = ()  # 用户不应当使用 state.name = "name" 来使用此类，因此我们限制用户不可自己对该类设置属性，但是可以通过属性访问字典；

    def __init__(self, *args, **kwargs):
        super(State, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item in self:
            _value = self[item]
            if isinstance(_value, dict):
                return State(_value)
            else:
                return _value
        else:
            raise ValueError(f"key '{item}' is not existed!")

@dataclass
class TrainerState:
    r"""
    该类用于我们 fastNLP 自己内部为了训练流程所记录的一些状态，当然是要暴露给用户给用户使用的；
    我们保存的state大部分上是 trainer 断点重训 需要重新加载的；
    专属于 `Trainer` 的状态记载的类；

    n_epochs: 训练过程中总共的 epoch 的数量；
    cur_epoch_idx: 当前正在运行第几个 epoch；
    global_forward_batches: 当前模型总共 forward 了多少个 step；
    batch_idx_in_epoch: 训练中在当前 epoch 的第几个 step；
    num_batches_per_epoch: 每一个 epoch 会 forward 多少个 step；
    total_batches: 完整训练过程会 forward 的 step 数量，注意 total_batches = total_batches * n_epochs；
    """
    n_epochs: Optional[int] = None  # 无论如何重新算

    cur_epoch_idx: Optional[int] = None  # 断点重训; 仅当 resume=False 时为0；
    global_forward_batches: Optional[int] = None  # 断点重训

    batch_idx_in_epoch: Optional[int] = None  # 断点重训

    num_batches_per_epoch: Optional[int] = None  # 无论如何重新算

    total_batches: Optional[int] = None  # 无论如何重新算

    def state_dict(self) -> Dict:
        r"""
        :return: 返回用于断点重训来保存的状态字典；
        """
        return {"cur_epoch_idx": self.cur_epoch_idx, "global_forward_batches": self.global_forward_batches,
                "batch_idx_in_epoch": self.batch_idx_in_epoch}

    def load_state_dict(self, state_dict: Dict):
        r"""
        用于断点重训来重新加载保存的状态字典；
        :param state_dict: 用于加载的状态字典；
        """
        for key in state_dict:
            assert key in {"cur_epoch_idx", "global_forward_batches", "batch_idx_in_epoch"}, "Wrong state_dict for `TrainerState`."
            setattr(self, key, state_dict[key])


