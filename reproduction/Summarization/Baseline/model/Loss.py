#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn.functional as F

from fastNLP.core.losses import LossBase
from fastNLP.core._logger import logger

class MyCrossEntropyLoss(LossBase):
    def __init__(self, pred=None, target=None, mask=None, padding_idx=-100, reduce='mean'):
        super().__init__()
        self._init_param_map(pred=pred, target=target, mask=mask)
        self.padding_idx = padding_idx
        self.reduce = reduce

    def get_loss(self, pred, target, mask):
        """
        
        :param pred: [batch, N, 2]
        :param target: [batch, N]
        :param input_mask: [batch, N]
        :return: 
        """
        # logger.debug(pred[0:5, :, :])
        # logger.debug(target[0:5, :])

        batch, N, _ = pred.size()
        pred = pred.view(-1, 2)
        target = target.view(-1)
        loss =  F.cross_entropy(input=pred, target=target,
                               ignore_index=self.padding_idx, reduction=self.reduce)
        loss = loss.view(batch, -1)
        loss = loss.masked_fill(mask.eq(0), 0)
        loss = loss.sum(1).mean()
        logger.debug("loss %f", loss)
        return loss


