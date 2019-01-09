import torch
import torch.nn as nn

from fastNLP.modules.encoder.lstm import LSTM
from fastNLP.modules.encoder.embedding import Embedding
from fastNLP.modules.encoder.linear import Linear
from fastNLP.modules.decoder.CRF import ConditionalRandomField
from fastNLP.modules.utils import seq_mask

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
        

    def forward(self, token_index_list, origin_len, speech_index_list=None):
        """

        :param pred: List of (torch.Tensor, or numpy.ndarray). Element's shape can be:
                torch.Size([B,]), torch.Size([B, n_classes]), torch.Size([B, max_len]), torch.Size([B, max_len, n_classes])
        :param target: List of (torch.Tensor, or numpy.ndarray). Element's can be:
                torch.Size([B,]), torch.Size([B,]), torch.Size([B, max_len]), torch.Size([B, max_len])
        :param seq_lens: List of (torch.Tensor, or numpy.ndarray). Element's can be:
                None, None, torch.Size([B], torch.Size([B]). ignored if masks are provided.
        :return: dict({'acc': float})
        """
        max_len = len(token_index_list[0])
        self.mask = self.make_mask(token_index_list, origin_len)
        
        x = self.Embedding(token_index_list) # [batch_size, max_len, word_emb_dim]
        x = self.Lstm(x) # [batch_size, max_len, hidden_size]
        x = self.Linear(x) # [batch_size, max_len, num_classes]
        
        loss = None
        ## Calculate the loss value if in the training mode(the speech_index_list is given)
        if speech_index_list is not None:
            total_loss = self.Crf(x, speech_index_list, self.mask) ## [batch_size, 1]
            loss = torch.mean(total_loss)
            
        
        ## Get the POS sequence(padding the sequence to equal length) 
        tag_seq = self.Crf.viterbi_decode(x, self.mask)
        for index in range(len(tag_seq)):
            bias = max_len - origin_len[index]
            for i in range(origin_len[index], max_len):
                tag_seq[index][i] = 0
        
        return {
            "loss": loss,
            "pred": tag_seq
        }
        
    
    def make_mask(self, x, seq_len):
        ## make the mask for batch-load datasets 
        batch_size, max_len = x.size(0), x.size(1)
        mask = seq_mask(seq_len, max_len)
        mask = mask.view(batch_size, max_len)
        mask = mask.to(x).float()
        return mask