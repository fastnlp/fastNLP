import numpy as np
import json
from os.path import join
import torch
import logging
import tempfile
import subprocess as sp
from datetime import timedelta
from time import time

from pyrouge import Rouge155
from pyrouge.utils import log

from fastNLP.core.losses import LossBase
from fastNLP.core.metrics import MetricBase

_ROUGE_PATH = '/path/to/RELEASE-1.5.5'

class MyBCELoss(LossBase):      
    
    def __init__(self, pred=None, target=None, mask=None):
        super(MyBCELoss, self).__init__()
        self._init_param_map(pred=pred, target=target, mask=mask)
        self.loss_func = torch.nn.BCELoss(reduction='none')

    def get_loss(self, pred, target, mask):
        loss = self.loss_func(pred, target.float())
        loss = (loss * mask.float()).sum()
        return loss

class LossMetric(MetricBase):
    def __init__(self, pred=None, target=None, mask=None):
        super(LossMetric, self).__init__()
        self._init_param_map(pred=pred, target=target, mask=mask)
        self.loss_func = torch.nn.BCELoss(reduction='none')
        self.avg_loss = 0.0
        self.nsamples = 0

    def evaluate(self, pred, target, mask):
        batch_size = pred.size(0)
        loss = self.loss_func(pred, target.float())
        loss = (loss * mask.float()).sum()
        self.avg_loss += loss
        self.nsamples += batch_size

    def get_metric(self, reset=True):
        self.avg_loss = self.avg_loss / self.nsamples
        eval_result = {'loss': self.avg_loss}
        if reset:
            self.avg_loss = 0
            self.nsamples = 0
        return eval_result
        
class RougeMetric(MetricBase):
    def __init__(self, data_path, dec_path, ref_path, n_total, n_ext=3, ngram_block=3, pred=None, target=None, mask=None):
        super(RougeMetric, self).__init__()
        self._init_param_map(pred=pred, target=target, mask=mask)
        self.data_path   = data_path
        self.dec_path    = dec_path
        self.ref_path    = ref_path
        self.n_total     = n_total
        self.n_ext       = n_ext
        self.ngram_block = ngram_block

        self.cur_idx = 0
        self.ext = []
        self.start = time()

    @staticmethod
    def eval_rouge(dec_dir, ref_dir):
        assert _ROUGE_PATH is not None
        log.get_global_console_logger().setLevel(logging.WARNING)
        dec_pattern = '(\d+).dec'
        ref_pattern = '#ID#.ref'
        cmd = '-c 95 -r 1000 -n 2 -m'
        with tempfile.TemporaryDirectory() as tmp_dir:
            Rouge155.convert_summaries_to_rouge_format(
                dec_dir, join(tmp_dir, 'dec'))
            Rouge155.convert_summaries_to_rouge_format(
                ref_dir, join(tmp_dir, 'ref'))
            Rouge155.write_config_static(
                join(tmp_dir, 'dec'), dec_pattern,
                join(tmp_dir, 'ref'), ref_pattern,
                join(tmp_dir, 'settings.xml'), system_id=1
            )
            cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
                + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
                + cmd
                + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
            output = sp.check_output(cmd.split(' '), universal_newlines=True)
            R_1 = float(output.split('\n')[3].split(' ')[3])
            R_2 = float(output.split('\n')[7].split(' ')[3])
            R_L = float(output.split('\n')[11].split(' ')[3])
            print(output)
        return R_1, R_2, R_L
    
    def evaluate(self, pred, target, mask):
        pred = pred + mask.float()
        pred = pred.cpu().data.numpy()
        ext_ids = np.argsort(-pred, 1)
        for sent_id in ext_ids:
            self.ext.append(sent_id)
        self.cur_idx += 1
        print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
              self.cur_idx, self.n_total, self.cur_idx/self.n_total*100, timedelta(seconds=int(time()-self.start))
             ), end='')

    def get_metric(self, use_ngram_block=True, reset=True):
        
        def check_n_gram(sentence, n, dic):
            tokens = sentence.split(' ')
            s_len = len(tokens)
            for i in range(s_len):
                if i + n > s_len:
                    break
                if ' '.join(tokens[i: i + n]) in dic:
                    return False
            return True # no n_gram overlap

        # load original data
        data = []
        with open(self.data_path) as f:
            for line in f:
                cur_data = json.loads(line)
                if 'text' in cur_data:
                    new_data = {}
                    new_data['article'] = cur_data['text']
                    new_data['abstract'] = cur_data['summary']
                    data.append(new_data)
                else:
                    data.append(cur_data)
        
        # write decode sentences and references
        if use_ngram_block == True:
            print('\nStart {}-gram blocking !!!'.format(self.ngram_block))
        for i, ext_ids in enumerate(self.ext):
            dec, ref = [], []
            if use_ngram_block == False:
                n_sent = min(len(data[i]['article']), self.n_ext)
                for j in range(n_sent):
                    idx = ext_ids[j]
                    dec.append(data[i]['article'][idx])
            else:
                n_sent = len(ext_ids)
                dic = {}
                for j in range(n_sent):
                    sent = data[i]['article'][ext_ids[j]]
                    if check_n_gram(sent, self.ngram_block, dic) == True:
                        dec.append(sent)
                        # update dic
                        tokens = sent.split(' ')
                        s_len = len(tokens)
                        for k in range(s_len):
                            if k + self.ngram_block > s_len:
                                break
                            dic[' '.join(tokens[k: k + self.ngram_block])] = 1
                        if len(dec) >= self.n_ext:
                            break

            for sent in data[i]['abstract']:
                ref.append(sent)

            with open(join(self.dec_path, '{}.dec'.format(i)), 'w') as f:
                for sent in dec:
                    print(sent, file=f)
            with open(join(self.ref_path, '{}.ref'.format(i)), 'w') as f:
                for sent in ref:
                    print(sent, file=f)
        
        print('\nStart evaluating ROUGE score !!!')
        R_1, R_2, R_L = RougeMetric.eval_rouge(self.dec_path, self.ref_path)
        eval_result = {'ROUGE-1': R_1, 'ROUGE-2': R_2, 'ROUGE-L':R_L}

        if reset == True:
            self.cur_idx = 0
            self.ext = []
            self.start = time()
        return eval_result
