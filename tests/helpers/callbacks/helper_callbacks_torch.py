import torch
from copy import deepcopy

from fastNLP.core.callbacks.callback import Callback


class RecordAccumulationStepsCallback_Torch(Callback):
    """
    通过该 callback 来测试 Trainer 的 accumulation_steps 是否实现正确；

    1. 在每一个 batch 检验模型是否正确地得到了更新（只有每隔 accumulation_steps 模型的参数才应该改变）；
    2. 检验 optimizer 的参数是否只在正确的时刻进行了清零；
    """

    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self.last_batch_params = None

        self.equal = 0

    def on_train_batch_end(self, trainer):
        # 注意这里的 trainer.global_forward_steps 的值比 trainer 上一次调用 batch_step_fn 的值大一；
        if trainer.global_forward_batches % trainer.accumulation_steps == 0:
            # 模型的参数应该与上一个 batch 不同；
            cur_batch_params = deepcopy(next(trainer.driver.unwrap_model().parameters()).cpu().detach())
            if self.last_batch_params is not None:
                assert not cur_batch_params.equal(self.last_batch_params)
                if cur_batch_params.equal(self.last_batch_params):
                    self.equal += 1

            # optimizer 的梯度应该得到了清零；
            optimizers = trainer.driver.optimizers
            for optimizer in optimizers:
                param_groups = optimizer.param_groups
                for group in param_groups:
                    for p in group['params']:
                        assert p.grad is None or p.grad.equal(torch.zeros_like(p.grad))
        else:
            # 模型的参数应该与上一个 batch 相同；
            cur_batch_params = deepcopy(next(trainer.driver.unwrap_model().parameters()).cpu().detach())
            if self.last_batch_params is not None:
                assert cur_batch_params.equal(self.last_batch_params)

            # optimizer 的梯度不应该得到了清零；
            optimizers = trainer.driver.optimizers
            for optimizer in optimizers:
                param_groups = optimizer.param_groups
                for group in param_groups:
                    for p in group['params']:
                        assert p.grad is not None and not p.grad.equal(torch.zeros_like(p.grad))

        self.last_batch_params = cur_batch_params

    def on_train_end(self, trainer):
        print(f"\n equal num: {self.equal}.\n")
        print(f"\ntotal_batch_num: {trainer.total_batches}.\n")
