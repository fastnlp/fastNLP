import time

import aggregation
import dataloader
import embedding
import encoder
import predict
import torch
import torch.nn as nn
import torch.optim as optim

WORD_NUM = 357361
WORD_SIZE = 100
HIDDEN_SIZE = 300
D_A = 350
R = 10
MLP_HIDDEN = 2000 
CLASSES_NUM = 5

from fastNLP.models.base_model import BaseModel


class MyNet(BaseModel):
    def __init__(self):
        super(MyNet, self).__init__()
        self.embedding = embedding.Lookuptable(WORD_NUM, WORD_SIZE)
        self.encoder = encoder.Lstm(WORD_SIZE, HIDDEN_SIZE, 1, 0.5, True)
        self.aggregation = aggregation.Selfattention(2 * HIDDEN_SIZE, D_A, R)
        self.predict = predict.MLP(R * HIDDEN_SIZE * 2, MLP_HIDDEN, CLASSES_NUM)
        self.penalty = None

    def encode(self, x):
        return self.encode(self.embedding(x))

    def aggregate(self, x):
        x, self.penalty = self.aggregate(x)
        return x

    def decode(self, x):
        return [self.predict(x), self.penalty]


class Net(nn.Module):
    """
    A model for sentiment analysis using lstm and self-attention
    """
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = embedding.Lookuptable(WORD_NUM, WORD_SIZE)
        self.encoder = encoder.Lstm(WORD_SIZE, HIDDEN_SIZE, 1, 0.5, True)
        self.aggregation = aggregation.Selfattention(2 * HIDDEN_SIZE, D_A, R)
        self.predict = predict.MLP(R * HIDDEN_SIZE * 2, MLP_HIDDEN, CLASSES_NUM)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x, penalty = self.aggregation(x)
        x = self.predict(x)
        return x, penalty


def train(model_dict=None, using_cuda=True, learning_rate=0.06,\
    momentum=0.3, batch_size=32, epochs=5, coef=1.0, interval=10):
    """
    training procedure

    Args: 
    If model_dict is given (a file address), it will continue training on the given model.
    Otherwise, it would train a new model from scratch.
    If using_cuda is true, the training would be conducted on GPU.
    Learning_rate and momentum is for SGD optimizer.
    coef is the coefficent between the cross-entropy loss and the penalization term.
    interval is the frequncy of reporting.

    the result will be saved with a form "model_dict_+current time", which could be used for further training
    """
    
    if using_cuda:
        net = Net().cuda()
    else:
        net = Net()
        
    if model_dict != None:
        net.load_state_dict(torch.load(model_dict))

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    dataset = dataloader.DataLoader("train_set.pkl", batch_size, using_cuda=using_cuda)

    #statistics
    loss_count = 0
    prepare_time = 0
    run_time = 0
    count = 0

    for epoch in range(epochs):
        print("epoch: %d"%(epoch))
        for i, batch in enumerate(dataset):
            t1 = time.time()
            X = batch["feature"]
            y = batch["class"]
            
            t2 = time.time()
            y_pred, y_penl = net(X)
            loss = criterion(y_pred, y) + torch.sum(y_penl) / batch_size * coef
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(net.parameters(), 0.5)
            optimizer.step()
            t3 = time.time()

            loss_count += torch.sum(y_penl).data[0]
            prepare_time += (t2 - t1)
            run_time += (t3 - t2)
            p, idx = torch.max(y_pred.data, dim=1)
            count += torch.sum(torch.eq(idx.cpu(), y.data.cpu()))

            if (i + 1) % interval == 0:
                print("epoch : %d, iters: %d"%(epoch, i + 1))     
                print("loss count:" + str(loss_count / (interval * batch_size)))
                print("acuracy:" + str(count / (interval * batch_size)))
                print("penalty:" + str(torch.sum(y_penl).data[0] / batch_size))
                print("prepare time:" + str(prepare_time))
                print("run time:" + str(run_time))
                prepare_time = 0
                run_time = 0
                loss_count = 0
                count = 0
        string = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        torch.save(net.state_dict(), "model_dict_%s.dict"%(string))

def test(model_dict, using_cuda=True):
    if using_cuda:
        net = Net().cuda()
    else:
        net = Net()
    net.load_state_dict(torch.load(model_dict))
    dataset = dataloader.DataLoader("test_set.pkl", batch_size=1, using_cuda=using_cuda)
    count = 0
    for i, batch in enumerate(dataset):
        X = batch["feature"]
        y = batch["class"]
        y_pred, _ = net(X)
        p, idx = torch.max(y_pred.data, dim=1)
        count += torch.sum(torch.eq(idx.cpu(), y.data.cpu()))
    print("accuracy: %f"%(count / dataset.num))
        

if __name__ == "__main__":
    train(using_cuda=torch.cuda.is_available())
    
    
    

