import torch
import torch.optim as optim


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        lr = self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
        # if step>self.warmup: lr = max(1e-4,lr)
        return lr


def get_std_opt(model):
    return NoamOpt(
        model.src_embed[0].d_model,
        2,
        4000,
        torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0,
            betas=(0.9, 0.98),
            eps=1e-9,
        ),
    )

