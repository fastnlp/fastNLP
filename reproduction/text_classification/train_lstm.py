import sys
sys.path.append('../..')

from fastNLP.io.pipe.classification import IMDBPipe
from fastNLP.embeddings import StaticEmbedding
from model.lstm import BiLSTMSentiment

from fastNLP import CrossEntropyLoss, AccuracyMetric
from fastNLP import Trainer
from torch.optim import Adam


class Config():
    train_epoch= 10
    lr=0.001

    num_classes=2
    hidden_dim=256
    num_layers=1
    nfc=128

    task_name = "IMDB"
    datapath={"train":"IMDB_data/train.csv", "test":"IMDB_data/test.csv"}
    save_model_path="./result_IMDB_test/"

opt=Config()


# load data
data_bundle=IMDBPipe.process_from_file(opt.datapath)

# print(data_bundle.datasets["train"])
# print(data_bundle)


# define model
vocab=data_bundle.vocabs['words']
embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-840b-300', requires_grad=True)
model=BiLSTMSentiment(init_embed=embed, num_classes=opt.num_classes, hidden_dim=opt.hidden_dim, num_layers=opt.num_layers, nfc=opt.nfc)


# define loss_function and metrics
loss=CrossEntropyLoss()
metrics=AccuracyMetric()
optimizer= Adam([param for param in model.parameters() if param.requires_grad==True], lr=opt.lr)


def train(data_bundle, model, optimizer, loss, metrics, opt):
    trainer = Trainer(data_bundle.datasets['train'], model, optimizer=optimizer, loss=loss,
                        metrics=metrics, dev_data=data_bundle.datasets['test'], device=0, check_code_level=-1,
                        n_epochs=opt.train_epoch, save_path=opt.save_model_path)
    trainer.train()


if __name__ == "__main__":
    train(data_bundle, model, optimizer, loss, metrics, opt)