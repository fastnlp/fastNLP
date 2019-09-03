from fastNLP.core.losses import LossBase

from reproduction.coreference_resolution.model.preprocess import get_labels
from reproduction.coreference_resolution.model.config import Config
import torch


class SoftmaxLoss(LossBase):
    """
    交叉熵loss
    允许多标签分类
    """

    def __init__(self, antecedent_scores=None, target=None, mention_start_tensor=None, mention_end_tensor=None):
        """

        :param pred:
        :param target:
        """
        super().__init__()
        self._init_param_map(antecedent_scores=antecedent_scores, target=target,
                             mention_start_tensor=mention_start_tensor, mention_end_tensor=mention_end_tensor)

    def get_loss(self, antecedent_scores, target, mention_start_tensor, mention_end_tensor):
        antecedent_labels = get_labels(target[0], mention_start_tensor, mention_end_tensor,
                                       Config().max_antecedents)

        antecedent_labels = torch.from_numpy(antecedent_labels*1).to(torch.device("cuda:" + Config().cuda))
        gold_scores = antecedent_scores + torch.log(antecedent_labels.float()).to(torch.device("cuda:" + Config().cuda))  # [num_mentions, max_ant + 1]
        marginalized_gold_scores = gold_scores.logsumexp(dim=1)  # [num_mentions]
        log_norm = antecedent_scores.logsumexp(dim=1)  # [num_mentions]
        return torch.sum(log_norm - marginalized_gold_scores)
