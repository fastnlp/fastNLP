import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def pack_sequence(tensor_seq, padding_value=0.0):
    if len(tensor_seq) <= 0:
        return
    length = [v.size(0) for v in tensor_seq]
    max_len = max(length)
    size = [len(tensor_seq), max_len]
    size.extend(list(tensor_seq[0].size()[1:]))
    ans = torch.Tensor(*size).fill_(padding_value)
    if tensor_seq[0].data.is_cuda:
        ans = ans.cuda()
    ans = Variable(ans)
    for i, v in enumerate(tensor_seq):
        ans[i, :length[i], :] = v
    return ans

class HAN(nn.Module):
    def __init__(self, input_size, output_size, 
                word_hidden_size, word_num_layers, word_context_size, 
                sent_hidden_size, sent_num_layers, sent_context_size):
        super(HAN, self).__init__()

        self.word_layer = AttentionNet(input_size, 
                                    word_hidden_size, 
                                    word_num_layers, 
                                    word_context_size)
        self.sent_layer = AttentionNet(2* word_hidden_size, 
                                    sent_hidden_size, 
                                    sent_num_layers, 
                                    sent_context_size)
        self.output_layer = nn.Linear(2* sent_hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, batch_doc):
        # input is a sequence of matrix
        doc_vec_list = []
        for doc in batch_doc:
            # doc's dim (num_sent, seq_len, word_dim)
            sent_mat = self.word_layer(doc)
            # sent_mat's dim (num_sent, vec_dim)
            doc_vec_list.append(sent_mat)
        doc_vec = self.sent_layer(pack_sequence(doc_vec_list))
        output = self.softmax(self.output_layer(doc_vec))
        return output

class AttentionNet(nn.Module):
    def __init__(self, input_size, gru_hidden_size, gru_num_layers, context_vec_size):
        super(AttentionNet, self).__init__()
        
        self.input_size = input_size
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.context_vec_size = context_vec_size

        # Encoder
        self.gru = nn.GRU(input_size=input_size, 
                        hidden_size=gru_hidden_size, 
                        num_layers=gru_num_layers, 
                        batch_first=True, 
                        bidirectional=True)
        # Attention
        self.fc = nn.Linear(2* gru_hidden_size, context_vec_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        # context vector
        self.context_vec = nn.Parameter(torch.Tensor(context_vec_size, 1))
        self.context_vec.data.uniform_(-0.1, 0.1)

    def forward(self, inputs):
        # inputs's dim (batch_size, seq_len,  word_dim)
        h_t, hidden = self.gru(inputs)
        u = self.tanh(self.fc(h_t))
        # u's dim (batch_size, seq_len, context_vec_size)
        alpha = self.softmax(torch.matmul(u, self.context_vec))
        # alpha's dim (batch_size, seq_len, 1)
        output = torch.bmm(torch.transpose(h_t, 1, 2), alpha)
        # output's dim (batch_size, 2*hidden_size, 1)
        return torch.squeeze(output, dim=2)


if __name__ == '__main__':
    import numpy as np
    use_cuda = True
    net = HAN(input_size=200, output_size=5, 
                word_hidden_size=50, word_num_layers=1, word_context_size=100,
                sent_hidden_size=50, sent_num_layers=1, sent_context_size=100)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.NLLLoss()
    test_time = 10
    batch_size = 64
    if use_cuda:
        net.cuda()
    print('test training')
    for step in range(test_time):
        x_data = [torch.randn(np.random.randint(1,10), 200, 200) for i in range(batch_size)]
        y_data = torch.LongTensor([np.random.randint(0, 5) for i in range(batch_size)])
        if use_cuda:
            x_data = [x_i.cuda() for x_i in x_data]
            y_data = y_data.cuda()
        x = [Variable(x_i) for x_i in x_data]
        y = Variable(y_data)
        predict = net(x)
        loss = criterion(predict, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.data[0])