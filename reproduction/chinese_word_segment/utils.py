
import torch


def seq_lens_to_mask(seq_lens):
    batch_size = seq_lens.size(0)
    max_len = seq_lens.max()

    indexes = torch.arange(max_len).view(1, -1).repeat(batch_size, 1).to(seq_lens.device)
    masks = indexes.lt(seq_lens.unsqueeze(1))

    return masks


def cut_long_training_sentences(sentences, max_sample_length=200):
    cutted_sentence = []
    for sent in sentences:
        sent_no_space = sent.replace(' ', '')
        if len(sent_no_space) > max_sample_length:
            parts = sent.strip().split()
            new_line = ''
            length = 0
            for part in parts:
                length += len(part)
                new_line += part + ' '
                if length > max_sample_length:
                    new_line = new_line[:-1]
                    cutted_sentence.append(new_line)
                    length = 0
                    new_line = ''
            if new_line != '':
                cutted_sentence.append(new_line[:-1])
        else:
            cutted_sentence.append(sent)
    return cutted_sentence


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