
import os

import torch.nn.functional as F

from fastNLP.loader.dataset_loader import ClassDatasetLoader as Dataset_loader
from fastNLP.loader.embed_loader import EmbedLoader as EmbedLoader
from fastNLP.loader.config_loader import ConfigSection
from fastNLP.loader.config_loader import ConfigLoader

from fastNLP.models.base_model import BaseModel

from fastNLP.core.preprocess import ClassPreprocess as Preprocess
from fastNLP.core.trainer   import ClassificationTrainer

from fastNLP.modules.encoder.embedding import Embedding as Embedding
from fastNLP.modules.encoder.lstm import Lstm
from fastNLP.modules.aggregation.self_attention import SelfAttention
from fastNLP.modules.decoder.MLP import MLP


train_data_path =  'small_train_data.txt'
dev_data_path = 'small_dev_data.txt'
# emb_path = 'glove.txt'

lstm_hidden_size = 300
embeding_size = 300
attention_unit = 350
attention_hops = 10
class_num = 5
nfc = 3000
### data load  ###
train_dataset = Dataset_loader(train_data_path)
train_data = train_dataset.load()

dev_args = Dataset_loader(dev_data_path)
dev_data = dev_args.load()

######  preprocess ####
preprocess = Preprocess()
word2index, label2index = preprocess.build_dict(train_data)
train_data, dev_data = preprocess.run(train_data, dev_data)

print("label2index:", label2index)

# emb = EmbedLoader(emb_path)
# embedding = emb.load_embedding(emb_dim= embeding_size , emb_file= emb_path ,word_dict= word2index)
### construct vocab ###

class SELF_ATTENTION_YELP_CLASSIFICATION(BaseModel):
    def __init__(self, args=None):
        super(SELF_ATTENTION_YELP_CLASSIFICATION,self).__init__()
        self.embedding = Embedding(len(word2index) ,embeding_size , init_emb= None )
        self.lstm = Lstm(input_size = embeding_size,hidden_size = lstm_hidden_size ,bidirectional = True)
        self.attention = SelfAttention(lstm_hidden_size * 2 ,dim =attention_unit ,num_vec=attention_hops)
        self.mlp = MLP(size_layer=[lstm_hidden_size * 2*attention_hops ,nfc ,class_num ] ,num_class=class_num  ,)
    def forward(self,x):
        x_emb = self.embedding(x)
        output = self.lstm(x_emb)
        after_attention, penalty = self.attention(output,x)
        after_attention =after_attention.view(after_attention.size(0),-1)
        output = self.mlp(after_attention)
        return output

    def loss(self, predict, ground_truth):
        print("predict:%s; g:%s" % (str(predict.size()), str(ground_truth.size())))
        print(ground_truth)
        return F.cross_entropy(predict, ground_truth)

train_args = ConfigSection()
ConfigLoader("good path").load_config('config.cfg',{"train": train_args})
train_args['vocab'] = len(word2index)


trainer = ClassificationTrainer(**train_args.data)

# for k in train_args.__dict__.keys():
#     print(k, train_args[k])
model = SELF_ATTENTION_YELP_CLASSIFICATION(train_args)
trainer.train(model,train_data , dev_data)
