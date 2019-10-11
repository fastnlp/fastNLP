from torch import nn
import torch
from reproduction.sequence_labelling.cws.model.module import FeatureFunMax, SemiCRFShiftRelay
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
    def __init__(self, char_embed, bigram_embed, hidden_size:int=400, num_layers:int=1, L:int=6, drop_p:float=0.2):
        super().__init__()
        self.char_embedding = char_embed
        self.bigram_embedding = bigram_embed
        self.lstm = LSTM(char_embed.embed_size+bigram_embed.embed_size, hidden_size // 2, num_layers=num_layers,
                         bidirectional=True,
                         batch_first=True)
        self.feature_fn = FeatureFunMax(hidden_size, L)
        self.semi_crf_relay = SemiCRFShiftRelay(L)
        self.feat_drop = nn.Dropout(drop_p)
        self.reset_param()

    def reset_param(self):
        for name, param in self.named_parameters():
            if 'embedding' in name:
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
        chars = self.char_embedding(chars)
        bigrams = self.bigram_embedding(bigrams)
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

