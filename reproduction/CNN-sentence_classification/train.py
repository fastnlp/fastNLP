import os

import
import
import torch
import torch.nn as nn
.dataset as dst
from .model import CNN_text
from torch.autograd import Variable

# Hyper Parameters
batch_size = 50
learning_rate = 0.0001
num_epochs = 20 
cuda = True


#split Dataset
dataset = dst.MRDataset()
length = len(dataset)

train_dataset = dataset[:int(0.9*length)]
test_dataset = dataset[int(0.9*length):]

train_dataset = dst.train_set(train_dataset)
test_dataset = dst.test_set(test_dataset)



# Data Loader 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
 
#cnn 

cnn = CNN_text(embed_num=len(dataset.word2id()), pretrained_embeddings=dataset.word_embeddings())
if cuda:
    cnn.cuda()


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# train and tests
best_acc = None

for epoch in range(num_epochs):
    # Train the Model
    cnn.train()
    for i, (sents,labels) in enumerate(train_loader):
        sents = Variable(sents)
        labels = Variable(labels)
        if cuda:
            sents = sents.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = cnn(sents)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

    # Test the Model
    cnn.eval()
    correct = 0
    total = 0
    for sents, labels in test_loader:
        sents = Variable(sents)
        if cuda:
            sents = sents.cuda()
            labels = labels.cuda()
        outputs = cnn(sents)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    acc = 100. * correct / total
    print('Test Accuracy: %f %%' % (acc))
    
    if best_acc is None or acc > best_acc:
        best_acc = acc
        if os.path.exists("models") is False:
            os.makedirs("models")
        torch.save(cnn.state_dict(), 'models/cnn.pkl')
    else:
        learning_rate = learning_rate * 0.8

print("Best Accuracy: %f %%" % best_acc)
print("Best Model: models/cnn.pkl")
