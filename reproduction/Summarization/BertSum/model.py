import torch
from torch import nn
from torch.nn import init

from fastNLP.modules.encoder.bert import BertModel


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, mask_cls):
        h = self.linear(inputs).squeeze(-1) # [batch_size, seq_len]
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class BertSum(nn.Module):
    
    def __init__(self, hidden_size=768):
        super(BertSum, self).__init__()
        
        self.hidden_size = hidden_size

        self.encoder = BertModel.from_pretrained('/path/to/uncased_L-12_H-768_A-12')
        self.decoder = Classifier(self.hidden_size)

    def forward(self, article, segment_id, cls_id):
         
        # print(article.device)
        # print(segment_id.device)
        # print(cls_id.device)

        input_mask = 1 - (article == 0).long()
        mask_cls = 1 - (cls_id == -1).long()
        assert input_mask.size() == article.size()
        assert mask_cls.size() == cls_id.size()

        bert_out = self.encoder(article, token_type_ids=segment_id, attention_mask=input_mask)
        bert_out = bert_out[0][-1] # last layer

        sent_emb = bert_out[torch.arange(bert_out.size(0)).unsqueeze(1), cls_id]
        sent_emb = sent_emb * mask_cls.unsqueeze(-1).float()
        assert sent_emb.size() == (article.size(0), cls_id.size(1), self.hidden_size) # [batch_size, seq_len, hidden_size]

        sent_scores = self.decoder(sent_emb, mask_cls) # [batch_size, seq_len]
        assert sent_scores.size() == (article.size(0), cls_id.size(1))

        return {'pred': sent_scores, 'mask': mask_cls}
