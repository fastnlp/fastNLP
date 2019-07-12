from torch import nn
import torch
from fastNLP.embeddings import Embedding
import numpy as np
from reproduction.seqence_labelling.cws.model.module import FeatureFunMax, SemiCRFShiftRelay
from fastNLP.modules import LSTM

class ShiftRelayCWSModel(nn.Module):
    """
    该模型可以用于进行分词操作
    包含两个方法，
        forward(chars, bigrams, seq_len) -> {'loss': batch_size,}
        predict(chars, bigrams) -> {'pred': batch_size x max_len, 'pred_mask': batch_size x max_len}
            pred是对当前segment的长度预测，pred_mask是仅在有预测的位置为1

    :param char_embed: 预训练的Embedding或者embedding的shape
    :param bigram_embed: 预训练的Embedding或者embedding的shape
    :param hidden_size: LSTM的隐藏层大小
    :param num_layers: LSTM的层数
    :param L: SemiCRFShiftRelay的segment大小
    :param num_bigram_per_char: 每个character对应的bigram的数量
    :param drop_p: Dropout的大小
    """
    def __init__(self, char_embed:Embedding, bigram_embed:Embedding, hidden_size:int=400, num_layers:int=1,
                 L:int=6, num_bigram_per_char:int=1, drop_p:float=0.2):
        super().__init__()
        self.char_embedding = Embedding(char_embed, dropout=drop_p)
        self._pretrained_embed = False
        if isinstance(char_embed, np.ndarray):
            self._pretrained_embed = True
        self.bigram_embedding = Embedding(bigram_embed, dropout=drop_p)
        self.lstm = LSTM(100 * (num_bigram_per_char + 1), hidden_size // 2, num_layers=num_layers, bidirectional=True,
                         batch_first=True)
        self.feature_fn = FeatureFunMax(hidden_size, L)
        self.semi_crf_relay = SemiCRFShiftRelay(L)
        self.feat_drop = nn.Dropout(drop_p)
        self.reset_param()
        # self.feature_fn.reset_parameters()

    def reset_param(self):
        for name, param in self.named_parameters():
            if 'embedding' in name and self._pretrained_embed:
                continue
            if 'bias_hh' in name:
                nn.init.constant_(param, 0)
            elif 'bias_ih' in name:
                nn.init.constant_(param, 1)
            elif len(param.size()) < 2:
                nn.init.uniform_(param, -0.1, 0.1)
            else:
                nn.init.xavier_uniform_(param)

    def get_feats(self, chars, bigrams, seq_len):
        batch_size, max_len = chars.size()
        chars = self.char_embedding(chars)
        bigrams = self.bigram_embedding(bigrams)
        bigrams = bigrams.view(bigrams.size(0), max_len, -1)
        chars = torch.cat([chars, bigrams], dim=-1)
        feats, _ = self.lstm(chars, seq_len)
        feats = self.feat_drop(feats)
        logits, relay_logits = self.feature_fn(feats)

        return logits, relay_logits

    def forward(self, chars, bigrams, relay_target, relay_mask, end_seg_mask, seq_len):
        logits, relay_logits = self.get_feats(chars, bigrams, seq_len)
        loss = self.semi_crf_relay(logits, relay_logits, relay_target, relay_mask, end_seg_mask, seq_len)
        return {'loss':loss}

    def predict(self, chars, bigrams, seq_len):
        logits, relay_logits = self.get_feats(chars, bigrams, seq_len)
        pred, pred_mask = self.semi_crf_relay.predict(logits, relay_logits, seq_len)
        return {'pred': pred, 'pred_mask': pred_mask}

