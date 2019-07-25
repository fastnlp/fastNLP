import torch
import torch.nn as nn
from fastNLP.core.const import Const as C
from fastNLP.modules.encoder.lstm import LSTM
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.modules.encoder.attention import SelfAttention
from fastNLP.modules.decoder.mlp import MLP


class BiLSTM_SELF_ATTENTION(nn.Module):
    def __init__(self, init_embed,
                 num_classes,
                 hidden_dim=256,
                 num_layers=1,
                 attention_unit=256,
                 attention_hops=1,
                 nfc=128):
        super(BiLSTM_SELF_ATTENTION,self).__init__()
        self.embed = get_embeddings(init_embed)
        self.lstm = LSTM(input_size=self.embed.embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)
        self.attention = SelfAttention(input_size=hidden_dim * 2 , attention_unit=attention_unit, attention_hops=attention_hops)
        self.mlp = MLP(size_layer=[hidden_dim* 2*attention_hops, nfc, num_classes])

    def forward(self, words):
        x_emb = self.embed(words)
        output, _ = self.lstm(x_emb)
        after_attention, penalty = self.attention(output,words)
        after_attention =after_attention.view(after_attention.size(0),-1)
        output = self.mlp(after_attention)
        return {C.OUTPUT: output}

    def predict(self, words):
        output = self(words)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}
