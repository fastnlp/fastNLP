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
from __future__ import division


import torch
from rouge import Rouge

from fastNLP.core.const import Const
from fastNLP.core.metrics import MetricBase

from tools.logger import *
from tools.utils import pyrouge_score_all, pyrouge_score_all_multi

class LabelFMetric(MetricBase):
    def __init__(self, pred=None, target=None):
        super().__init__()

        self._init_param_map(pred=pred, target=target)

        self.match = 0.0
        self.pred = 0.0
        self.true = 0.0
        self.match_true = 0.0
        self.total = 0.0


    def evaluate(self, pred, target):
        """
        
        :param pred: [batch, N] int
        :param target: [batch, N] int
        :return: 
        """
        target = target.data
        pred = pred.data
        # logger.debug(pred.size())
        # logger.debug(pred[:5,:])
        batch, N = pred.size()
        self.pred += pred.sum()
        self.true += target.sum()
        self.match += (pred == target).sum()
        self.match_true += ((pred == target) & (pred == 1)).sum()
        self.total += batch * N

    def get_metric(self, reset=True):
        self.match,self.pred, self.true, self.match_true, self.total = self.match.float(),self.pred.float(), self.true.float(), self.match_true.float(), self.total
        logger.debug((self.match,self.pred, self.true, self.match_true, self.total))
        try:
            accu = self.match / self.total
            precision = self.match_true / self.pred
            recall = self.match_true / self.true
            F = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            F = 0.0
            logger.error("[Error] float division by zero")
        if reset:
            self.pred, self.true, self.match_true, self.match, self.total = 0, 0, 0, 0, 0
        ret = {"accu": accu.cpu(), "p":precision.cpu(), "r":recall.cpu(), "f": F.cpu()}
        logger.info(ret)
        return ret


class RougeMetric(MetricBase):
    def __init__(self, hps, pred=None, text=None, refer=None):
        super().__init__()

        self._hps = hps
        self._init_param_map(pred=pred, text=text, summary=refer)

        self.hyps = []
        self.refers = []

    def evaluate(self, pred, text, summary):
        """

        :param prediction: [batch, N]
        :param text: [batch, N]
        :param summary: [batch, N]
        :return: 
        """

        batch_size, N = pred.size()
        for j in range(batch_size):
            original_article_sents = text[j]
            sent_max_number = len(original_article_sents)
            refer = "\n".join(summary[j])
            hyps = "\n".join(original_article_sents[id] for id in range(len(pred[j])) if
                             pred[j][id] == 1 and id < sent_max_number)
            if sent_max_number < self._hps.m and len(hyps) <= 1:
                print("sent_max_number is too short %d, Skip!", sent_max_number)
                continue

            if len(hyps) >= 1 and hyps != '.':
                self.hyps.append(hyps)
                self.refers.append(refer)
            elif refer == "." or refer == "":
                logger.error("Refer is None!")
                logger.debug(refer)
            elif hyps == "." or hyps == "":
                logger.error("hyps is None!")
                logger.debug("sent_max_number:%d", sent_max_number)
                logger.debug("pred:")
                logger.debug(pred[j])
                logger.debug(hyps)
            else:
                logger.error("Do not select any sentences!")
                logger.debug("sent_max_number:%d", sent_max_number)
                logger.debug(original_article_sents)
                logger.debug(refer)
                continue

    def get_metric(self, reset=True):
        pass

class FastRougeMetric(RougeMetric):
    def __init__(self, hps, pred=None, text=None, refer=None):
        super().__init__(hps, pred, text, refer)

    def get_metric(self, reset=True):
        logger.info("[INFO] Hyps and Refer number is %d, %d", len(self.hyps), len(self.refers))
        if len(self.hyps) == 0 or len(self.refers) == 0 :
            logger.error("During testing, no hyps or refers is selected!")
            return
        rouge = Rouge()
        scores_all = rouge.get_scores(self.hyps, self.refers, avg=True)
        if reset:
            self.hyps = []
            self.refers = []
        logger.info(scores_all)
        return scores_all


class PyRougeMetric(RougeMetric):
    def __init__(self, hps, pred=None, text=None, refer=None):
        super().__init__(hps, pred, text, refer)

    def get_metric(self, reset=True):
        logger.info("[INFO] Hyps and Refer number is %d, %d", len(self.hyps), len(self.refers))
        if len(self.hyps) == 0 or len(self.refers) == 0:
            logger.error("During testing, no hyps or refers is selected!")
            return
        if isinstance(self.refers[0], list):
            logger.info("Multi Reference summaries!")
            scores_all = pyrouge_score_all_multi(self.hyps, self.refers)
        else:
            scores_all = pyrouge_score_all(self.hyps, self.refers)
        if reset:
            self.hyps = []
            self.refers = []
        logger.info(scores_all)
        return scores_all



