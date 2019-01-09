from fastNLP.core.metrics import MetricBase
import torch
import numpy as np

class PosMetric(MetricBase):
    """
        The PosMetric use the accuracy of each word to 
        evaluate the performance of POS task, suggested by the 
        original paper on https://arxiv.org/abs/1508.01991
    """
    def __init__(self, pred=None, target=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target)
        self.total = 0
        self.acc_count = 0
        

    def evaluate1(self, pred, target):
        """
            Each time when loading a batch of data in the Trainer&Tester, 
            this function would be called for one time. So we can use some 
            class member to memorize the state in the training process.
        
        """
        self.acc_count += torch.sum(torch.eq(pred, target).float()).item()
        self.total += np.prod(list(pred.size()))

    def evaluate(self, pred, target):  
        
        for i in range(len(pred)):
            for j in range(len(pred[0])):
                if target[i][j] != 0:
                    self.acc_count += 1 if target[i][j] == pred[i][j] else 0
                    self.total += 1
    
    def get_metric(self):
        """
            As suggested in the tutorial, this function would be called once 
            the Trainer finished 1 epoch of training on the whole dataset.
            
            :return {"acc": float}
        """
        
        return {
            'acc': round(self.acc_count / self.total, 6)
        }