import torch
import torch.nn as nn
from fastNLP.core.const import Const as C
from .awdlstm_module import LSTM
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.modules.decoder.mlp import MLP


class AWDLSTMSentiment(nn.Module):
    def __init__(self, init_embed,
                 num_classes,
                 hidden_dim=256,
                 num_layers=1,
                 nfc=128,
                 wdrop=0.5):
        super(AWDLSTMSentiment,self).__init__()
        self.embed = get_embeddings(init_embed)
        self.lstm = LSTM(input_size=self.embed.embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True, wdrop=wdrop)
        self.mlp = MLP(size_layer=[hidden_dim* 2, nfc, num_classes])

    def forward(self, words):
        x_emb = self.embed(words)
        output, _ = self.lstm(x_emb)
        output = self.mlp(output[:,-1,:])
        return {C.OUTPUT: output}

    def predict(self, words):
        output = self(words)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}

