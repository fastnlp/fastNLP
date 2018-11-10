
import torch


def seq_lens_to_mask(seq_lens):
    batch_size = seq_lens.size(0)
    max_len = seq_lens.max()

    indexes = torch.arange(max_len).view(1, -1).repeat(batch_size, 1).to(seq_lens.device)
    masks = indexes.lt(seq_lens.unsqueeze(1))

    return masks


from itertools import chain

def refine_ys_on_seq_len(ys, seq_lens):
    refined_ys = []
    for b_idx, length in enumerate(seq_lens):
        refined_ys.append(list(ys[b_idx][:length]))

    return refined_ys

def flat_nested_list(nested_list):
    return list(chain(*nested_list))

def calculate_pre_rec_f1(model, batcher):
    true_ys, pred_ys = decode_iterator(model, batcher)

    true_ys = flat_nested_list(true_ys)
    pred_ys = flat_nested_list(pred_ys)

    cor_num = 0
    yp_wordnum = pred_ys.count(1)
    yt_wordnum = true_ys.count(1)
    start = 0
    for i in range(len(true_ys)):
        if true_ys[i] == 1:
            flag = True
            for j in range(start, i + 1):
                if true_ys[j] != pred_ys[j]:
                    flag = False
                    break
            if flag:
                cor_num += 1
            start = i + 1
    P = cor_num / (float(yp_wordnum) + 1e-6)
    R = cor_num / (float(yt_wordnum) + 1e-6)
    F = 2 * P * R / (P + R + 1e-6)
    return P, R, F


def decode_iterator(model, batcher):
    true_ys = []
    pred_ys = []
    seq_lens = []
    with torch.no_grad():
        model.eval()
        for batch_x, batch_y in batcher:
            pred_dict = model(batch_x)
            seq_len = pred_dict['seq_lens'].cpu().numpy()
            probs = pred_dict['pred_probs']
            _, pred_y = probs.max(dim=-1)
            true_y = batch_y['tags']
            pred_y = pred_y.cpu().numpy()
            true_y = true_y.cpu().numpy()

            true_ys.extend(list(true_y))
            pred_ys.extend(list(pred_y))
            seq_lens.extend(list(seq_len))
        model.train()

    true_ys = refine_ys_on_seq_len(true_ys, seq_lens)
    pred_ys = refine_ys_on_seq_len(pred_ys, seq_lens)

    return true_ys, pred_ys


from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, gamma=2, size_average=True, reduce=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=-1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask.requires_grad = True
        ids = targets.view(-1, 1)
        class_mask = class_mask.scatter(1, ids.data, 1.)

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
        if self.reduce:
            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss.sum()
            return loss
        return batch_loss