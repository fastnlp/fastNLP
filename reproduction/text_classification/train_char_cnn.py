import sys
sys.path.append('../..')
from fastNLP.core.const import Const as C
import torch.nn as nn
from fastNLP.io.pipe.classification import YelpFullPipe,YelpPolarityPipe,SST2Pipe,IMDBPipe
from model.char_cnn import CharacterLevelCNN
from fastNLP import CrossEntropyLoss, AccuracyMetric
from fastNLP.core.trainer import Trainer
from torch.optim import SGD
from torch.autograd import Variable
import torch
from torch.optim.lr_scheduler import LambdaLR
from fastNLP.core import LRScheduler


##hyper
#todo 这里加入fastnlp的记录
class Config():
    #seed=7777
    model_dir_or_name="en-base-uncased"
    embedding_grad= False,
    bert_embedding_larers= '4,-2,-1'
    train_epoch= 100
    num_classes=2
    task= "yelp_p"
    lr=0.01
    batch_size=128
    model_size="large"
    number_of_characters=69
    extra_characters=''
    max_length=1014
    weight_decay = 1e-5
    to_lower=True
    tokenizer = 'spacy'  # 使用spacy进行分词

    char_cnn_config={
        "alphabet": {
            "en": {
                "lower": {
                    "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                    "number_of_characters": 69
                },
                "both": {
                    "alphabet": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                    "number_of_characters": 95
                }
            }
        },
        "model_parameters": {
            "small": {
                "conv": [
                    #依次是channel，kennnel_size，maxpooling_size
                    [256,7,3],
                    [256,7,3],
                    [256,3,-1],
                    [256,3,-1],
                    [256,3,-1],
                    [256,3,3]
                ],
                "fc": [1024,1024]
            },
            "large":{
                "conv":[
                    [1024, 7, 3],
                    [1024, 7, 3],
                    [1024, 3, -1],
                    [1024, 3, -1],
                    [1024, 3, -1],
                    [1024, 3, 3]
                ],
                "fc": [2048,2048]
            }
        },
        "data": {
            "text_column": "SentimentText",
            "label_column": "Sentiment",
            "max_length": 1014,
            "num_of_classes": 2,
            "encoding": None,
            "chunksize": 50000,
            "max_rows": 100000,
            "preprocessing_steps": ["lower", "remove_hashtags", "remove_urls", "remove_user_mentions"]
        },
        "training": {
            "batch_size": 128,
            "learning_rate": 0.01,
            "epochs": 10,
            "optimizer": "sgd"
        }
    }
ops=Config

# set_rng_seeds(ops.seed)
# print('RNG SEED: {}'.format(ops.seed))


##1.task相关信息：利用dataloader载入DataBundle
char_vocab=ops.char_cnn_config["alphabet"]["en"]["lower"]["alphabet"]
ops.number_of_characters=len(char_vocab)
ops.embedding_dim=ops.number_of_characters

# load data set
if ops.task == 'yelp_p':
    data_bundle = YelpPolarityPipe(lower=ops.to_lower, tokenizer=ops.tokenizer).process_from_file()
elif ops.task == 'yelp_f':
    data_bundle = YelpFullPipe(lower=ops.to_lower, tokenizer=ops.tokenizer).process_from_file()
elif ops.task == 'imdb':
    data_bundle = IMDBPipe(lower=ops.to_lower, tokenizer=ops.tokenizer).process_from_file()
elif ops.task == 'sst-2':
    data_bundle = SST2Pipe(lower=ops.to_lower, tokenizer=ops.tokenizer).process_from_file()
else:
    raise RuntimeError(f'NOT support {ops.task} task yet!')

print(data_bundle)

def wordtochar(words):
    chars = []

    #for word in words:
        #word = word.lower()
    for char in words:
        chars.append(char)
        #chars.append('')
    #chars.pop()
    return chars

#chartoindex
def chartoindex(chars):
    max_seq_len=ops.max_length
    zero_index=len(char_vocab)
    char_index_list=[]
    for char in chars:
        if char in char_vocab:
             char_index_list.append(char_vocab.index(char))
        else:
            #<unk>和<pad>均使用最后一个作为embbeding
            char_index_list.append(zero_index)
    if len(char_index_list) > max_seq_len:
        char_index_list = char_index_list[:max_seq_len]
    elif 0 < len(char_index_list) < max_seq_len:
        char_index_list = char_index_list+[zero_index]*(max_seq_len-len(char_index_list))
    elif len(char_index_list) == 0:
        char_index_list=[zero_index]*max_seq_len
    return char_index_list


for dataset in data_bundle.datasets.values():
    dataset.apply_field(wordtochar, field_name="raw_words", new_field_name='chars')
    dataset.apply_field(chartoindex,field_name='chars',new_field_name='chars')

# print(data_bundle.datasets['train'][0]['chars'])
# print(data_bundle.datasets['train'][0]['raw_words'])

data_bundle.set_input('chars')
data_bundle.set_target('target')

##2. 定义/组装模型，这里可以随意，就如果是fastNLP封装好的，类似CNNText就直接用初始化调用就好了，这里只是给出一个伪框架表示占位，在这里建立符合fastNLP输入输出规范的model
class ModelFactory(nn.Module):
    """
        用于拼装embedding，encoder，decoder 以及设计forward过程

        :param embedding:  embbeding model
        :param encoder: encoder model
        :param decoder: decoder model

        """
    def __int__(self,embedding,encoder,decoder,**kwargs):
        super(ModelFactory,self).__init__()
        self.embedding=embedding
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,x):
        return {C.OUTPUT:None}

## 2.或直接复用fastNLP的模型
vocab_label=data_bundle.get_vocab('target')
'''
# emded_char=CNNCharEmbedding(vocab)
# embed_word = StaticEmbedding(vocab, model_dir_or_name='en-glove-6b-50', requires_grad=True)
# embedding=StackEmbedding([emded_char, embed_word])
# cnn_char_embed = CNNCharEmbedding(vocab)
# lstm_char_embed = LSTMCharEmbedding(vocab)
# embedding = StackEmbedding([cnn_char_embed, lstm_char_embed])
'''
#one-hot embedding
embedding_weight= Variable(torch.zeros(len(char_vocab)+1, len(char_vocab)))

for i in range(len(char_vocab)):
    embedding_weight[i][i]=1
embedding=nn.Embedding(num_embeddings=len(char_vocab)+1,embedding_dim=len(char_vocab),padding_idx=len(char_vocab),_weight=embedding_weight)
for para in embedding.parameters():
    para.requires_grad=False
#CNNText太过于简单
#model=CNNText(embed=embedding, num_classes=ops.num_classes)
model=CharacterLevelCNN(ops,embedding)

## 3. 声明loss,metric,optimizer
loss=CrossEntropyLoss
metric=AccuracyMetric
optimizer = SGD([param for param in model.parameters() if param.requires_grad == True],
                lr=ops.lr, momentum=0.9, weight_decay=ops.weight_decay)
callbacks = []
# callbacks.append(LRScheduler(CosineAnnealingLR(optimizer, 5)))
callbacks.append(
    LRScheduler(LambdaLR(optimizer, lambda epoch: ops.lr if epoch <
                         ops.train_epoch * 0.8 else ops.lr * 0.1))
)

## 4.定义train方法
def train(model,datainfo,loss,metrics,optimizer,num_epochs=100):
    trainer = Trainer(datainfo.datasets['train'], model, optimizer=optimizer, loss=loss(target='target'),batch_size=ops.batch_size,
                      metrics=[metrics(target='target')], dev_data=datainfo.datasets['test'], device=[0,1,2], check_code_level=-1,
                      n_epochs=num_epochs,callbacks=callbacks)
    print(trainer.train())



if __name__=="__main__":
    train(model,data_bundle,loss,metric,optimizer,num_epochs=ops.train_epoch)
