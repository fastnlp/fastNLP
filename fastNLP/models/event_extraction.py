import torch
from fastNLP.models.sequence_modeling import SeqLabeling,BaseModel
from fastNLP.modules import decoder, encoder
class Trigger_extraction(SeqLabeling):
    """
    PyTorch Network for trigger extraction
    """

    def __init__(self, args,emb=None):
        super(Trigger_extraction,self).__init__(args)
        vocab_size = args["vocab_size"]
        word_emb_dim = args["word_emb_dim"]
        hidden_dim = args["rnn_hidden_units"]
        num_classes = args["num_classes"]

        self.Embedding = encoder.embedding.Embedding(vocab_size, word_emb_dim, init_emb=emb)
        self.Rnn = encoder.lstm.Lstm(word_emb_dim, hidden_dim, num_layers=1, dropout=0.2, bidirectional=True)
        self.Linear1 = encoder.Linear(hidden_dim * 2, hidden_dim)
        self.batch_norm = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.5)
        self.Linear2 = encoder.Linear(hidden_dim, num_classes)

        self.Crf = decoder.CRF.ConditionalRandomField(num_classes)


    def forward(self, x):
        """
        :param x: LongTensor, [batch_size, mex_len]
        :return y: [batch_size, mex_len, tag_size]
        """

        batch_size = x.size(0)
        max_len = x.size(1)
        x = self.Embedding(x)
        x = self.drop(x)
        # [batch_size, max_len, word_emb_dim]
        x = self.Rnn(x)
        # [batch_size, max_len, hidden_size * direction]
        x = x.contiguous()
        x = x.view(batch_size * max_len, -1)
        x = self.Linear1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.Linear2(x)
        x = x.view(batch_size, max_len, -1)
        # [batch_size, max_len, num_classes]
        return x

class Argument_extraction(BaseModel):
    """
        PyTorch Network for argument extraction
    """
    def __init__(self, args,emb=None):
        super(Argument_extraction,self).__init__()
        vocab_size = args["vocab_size"]
        word_emb_dim = args["word_emb_dim"]
        hidden_dim = args["rnn_hidden_units"]
        num_classes = args["num_classes"]

        self.Embedding = encoder.embedding.Embedding(vocab_size, word_emb_dim, init_emb=emb)
        self.Rnn = encoder.lstm.Lstm(word_emb_dim, hidden_dim, num_layers=1, dropout=0.2, bidirectional=True)
        self.Linear1 = encoder.Linear(hidden_dim * 4, hidden_dim)
        self.batch_norm = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.5)
        self.Linear2 = encoder.Linear(hidden_dim, num_classes)


        self.CrossEn= torch.nn.CrossEntropyLoss(reduce=False)

    def forward(self, x, entity, tr_ph):
        """
        :param x: LongTensor, [batch_size, mex_len]
        :return y: [batch_size, mex_len, tag_size]
        """

        batch_size = x.size(0)
        max_len = x.size(1)
        x = self.Embedding(x)
        x = self.drop(x)
        x = self.Rnn(x)
        if x.is_cuda:
            tr_ph_one_hot=torch.zeros(batch_size,max_len).long().cuda().scatter_(1,tr_ph.view(batch_size,1),1)
        else:
            tr_ph_one_hot = torch.zeros(batch_size, max_len).long().scatter_(1, tr_ph.view(batch_size,1), 1)
        ph=tr_ph_one_hot.view(batch_size,max_len,1)
        ph=ph.expand(-1,-1,x.size(2))
        value_=torch.gather(x,1,ph)
        x=torch.cat([x,value_],-1)
        # [batch_size, max_len, hidden_size * direction]
        x = x.contiguous()
        x = x.view(batch_size * max_len, -1)
        x = self.Linear1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.Linear2(x)
        x = x.view(batch_size, max_len, -1)
        return x

    def loss(self,x,y,mask):
        x = x.float()
        y = y.long()
        batch_size=x.size(0)
        max_len=x.size(1)
        dim=x.size(2)
        x_input=x.view(batch_size*max_len,dim)
        y_input=y.view(batch_size*max_len)
        mask=mask.float().view(batch_size*max_len)
        loss=self.CrossEn(x_input,y_input)
        loss=loss*mask
        loss=loss.view(batch_size,max_len)
        return torch.mean(torch.sum(loss,-1))

    def prediction(self,x,mask):
        x=x.float()
        mask=mask.long()
        x=torch.argmax(x,-1)
        return x*mask




