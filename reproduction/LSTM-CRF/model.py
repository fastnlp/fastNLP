import torch
import torch.nn as nn

from fastNLP.modules.encoder.embedding import Embedding
from fastNLP.modules.decoder.CRF import ConditionalRandomField
from fastNLP.modules.utils import seq_mask
from fastNLP.modules.encoder.lstm import LSTM

class BiLSTMCRF(nn.Module):
    
    def __init__(self, config):
        super(BiLSTMCRF, self).__init__()
        vocab_size = config["vocab_size"]
        word_emb_dim = config["word_emb_dim"]
        hidden_dim = config["rnn_hidden_units"]
        num_classes = config["num_classes"]
        bi_direciton = config["bi_direction"]
        self.Embedding = Embedding(vocab_size, word_emb_dim)
        self.Lstm = LSTM(word_emb_dim, hidden_dim, bidirectional=bi_direciton)
        self.Linear = Linear(2*hidden_dim if bi_direciton else hidden_dim, num_classes)
        self.Crf = ConditionalRandomField(num_classes)
        self.mask = None
        

    def forward(self, token_index_list, speech_index_list=None):
        max_len = len(token_index_list)
        self.mask = self.make_mask(token_index_list, max_len)
        
        x = self.Embedding(token_index_list) # [batch_size, max_len, word_emb_dim]
        x = self.Lstm(x) # [batch_size, max_len, hidden_size]
        x = self.Linear(x) # [batch_size, max_len, num_classes]
        
        loss = None
        ## Calculate the loss value
        if speech_index_list is not None:
            total_loss = self.Crf(x, speech_index_list, self.mask) ## [batch_size, 1]
            loss = torch.mean(total_loss)
        ## Get the part of speech
        tag_seq = self.Crf.viterbi_decode(x, self.mask)
        
        # pad prediction to equal length
#         import pdb;pdb.set_trace()
        for pred in tag_seq:
            if len(pred) < max_len:
                pred = torch.cat((pred, torch.LongTensor([0] * (max_len - len(pred))).cuda()))
        
        return {
            "loss": loss,
            "pred": tag_seq
        }
        
    
    def make_mask(self, x, seq_len):
        batch_size, max_len = x.size(0), x.size(1)
        mask = seq_mask(seq_len, max_len)
        mask = mask.view(batch_size, max_len)
        mask = mask.to(x).float()
        return mask