from typing import List
from fastNLP.envs.imports import _NEED_IMPORT_FAIRSCALE
if _NEED_IMPORT_FAIRSCALE:
    import torch
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
    from fairscale.optim import OSS

__all__ = [
    'ShardedDriver'
]

from .ddp import TorchDDPDriver


# todo 注意 fairscale 现在几乎所有的功能都没有实现；
# TODO：预跑前后对模型和 optimizers 的支持；
# TODO：fairscale 的 fp16 额外的处理；
class ShardedDriver(TorchDDPDriver):
    _REDUCE_BUFFER_SIZE_DEFAULT: int = 2 ** 23  # 8M

    def __init__(
            self,
            model,
            parallel_device: List["torch.device"],
            num_nodes: int = 1,
            fp16: bool = False,
            **kwargs
    ):
        super(ShardedDriver, self).__init__(
            model=model,
            parallel_device=parallel_device,
            num_nodes=num_nodes,
            fp16=fp16,
            **kwargs
        )

    def configure_ddp(self):
        if "reduce_buffer_size" not in self._ddp_kwargs:
            # For multi-node training, enabling bucketing will improve performance.
            self._ddp_kwargs["reduce_buffer_size"] = self._REDUCE_BUFFER_SIZE_DEFAULT if self.num_nodes > 1 else 0

        self.optimizers = self._wrap_optimizers(self.optimizers)
        self.model = ShardedDataParallel(self.model, sharded_optimizer=self.optimizers, **self._ddp_kwargs)


    def _wrap_optimizers(self, optimizers) -> List["OSS"]:
        # TODO：之后得去研究一下 pytorch lightning 为什么这样写，我们是不是也需要这样写；
        # if self.model is not None and self.model.trainer.state.fn != TrainerFn.FITTING:
        #     return optimizers

        return self._reinit_optimizers_with_oss(optimizers)

    def _reinit_optimizers_with_oss(self, optimizers) -> List["OSS"]:
        for x, optimizer in enumerate(optimizers):
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)

                # TODO：具体细节见 pytorch lightning 的这一函数，主要的点在于加入 fp16 相关的一些东西；
                optimizers[x] = zero_optimizer
                del optimizer
        return optimizers

