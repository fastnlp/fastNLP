from fastNLP.core.metrics import MetricBase

class PosMetric(MetricBase):
    def __init__(self, pred=None, target=None):
        super().__init__()

        self._init_param_map(pred=pred, target=target)

        self.total = 0
        self.acc_count = 0
        

    def evaluate(self, pred, target):
        """

        :param pred: List of (torch.Tensor, or numpy.ndarray). Element's shape can be:
                torch.Size([B,]), torch.Size([B, n_classes]), torch.Size([B, max_len]), torch.Size([B, max_len, n_classes])
        :param target: List of (torch.Tensor, or numpy.ndarray). Element's can be:
                torch.Size([B,]), torch.Size([B,]), torch.Size([B, max_len]), torch.Size([B, max_len])
        :param seq_lens: List of (torch.Tensor, or numpy.ndarray). Element's can be:
                None, None, torch.Size([B], torch.Size([B]). ignored if masks are provided.
        :return: dict({'acc': float})
        """
        
        self.acc_count += torch.sum(torch.eq(pred, target).float()).item()
        self.total += np.prod(list(pred.size()))

    def get_metric(self):
        return {
            'acc': round(self.acc_count / self.total, 6)
        }