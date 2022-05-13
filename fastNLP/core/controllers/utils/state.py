from dataclasses import dataclass
from typing import Optional, Dict

__all__ = [
    'State',
    'TrainerState'
]


class State(dict):
    r"""
    提供给用户使用的 ``state``，用来记载您的 ``callback`` 实时数据，该 ``state`` 实际上是一个字典，我们通过复用 ``__getattr__`` 方法来实现类似
    类属性的字典调用方式；

    为了实现断点重训，用户应当保证其保存的信息都是可序列化的；

    推荐的使用方式::

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

    :param n_epochs: 训练过程中总共的 epoch 的数量；
    :param cur_epoch_idx: 当前正在运行第几个 epoch；
    :param global_forward_batches: 当前模型总共 forward 了多少个 step；
    :param batch_idx_in_epoch: 训练中在当前 epoch 的第几个 step；
    :param num_batches_per_epoch: 每一个 epoch 会 forward 多少个 step；
    :param total_batches: 完整训练过程会 forward 的 step 数量，注意 total_batches = total_batches * n_epochs；
    """
    n_epochs: Optional[int] = None  # 无论如何重新算

    cur_epoch_idx: Optional[int] = 0  # 断点重训; 仅当 resume=False 时为0；
    global_forward_batches: Optional[int] = 0  # 断点重训

    batch_idx_in_epoch: Optional[int] = 0  # 断点重训

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


