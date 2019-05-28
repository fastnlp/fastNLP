import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastNLP.core.losses import LossBase


reduce_func = {
    'none': lambda x, mask: x*mask,
    'sum': lambda x, mask: (x*mask).sum(),
    'mean': lambda x, mask: (x*mask).sum() / mask.sum(),
}


class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=-100, reduction='mean'):
        global reduce_func
        super().__init__()
        if smoothing < 0 or smoothing > 1:
            raise ValueError('invalid smoothing value: {}'.format(smoothing))
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        if reduction not in reduce_func:
            raise ValueError('invalid reduce type: {}'.format(reduction))
        self.reduce_func = reduce_func[reduction]

    def forward(self, input, target):
        input = F.log_softmax(input, dim=1)  # [N, C, ...]
        smooth_val = self.smoothing / input.size(1)  # [N, C, ...]
        target_logit = input.new_full(input.size(), fill_value=smooth_val)
        target_logit.scatter_(1, target[:, None], 1 - self.smoothing)
        result = -(target_logit * input).sum(1)  # [N, ...]
        mask = (target != self.ignore_index).float()
        return self.reduce_func(result, mask)


class SmoothCE(LossBase):
    def __init__(self, pred=None, target=None, **kwargs):
        super().__init__()
        self.loss_fn = LabelSmoothCrossEntropy(**kwargs)
        self._init_param_map(pred=pred, target=target)

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)


if __name__ == '__main__':
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    sm_loss_fn = LabelSmoothCrossEntropy(smoothing=0, ignore_index=0)
    predict = torch.tensor([[0, 0.2, 0.7, 0.1, 0],
                            [0, 0.9, 0.2, 0.1, 0],
                            [1, 0.2, 0.7, 0.1, 0]])
    target = torch.tensor([2, 1, 0])
    loss = loss_fn(predict, target)
    sm_loss = sm_loss_fn(predict, target)
    print(loss, sm_loss)
