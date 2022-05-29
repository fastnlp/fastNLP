import os
import sys
import time
# os.environ["cuda_archs"] = "61"
# os.environ["FAS"]
os.environ["log_silent"] = "1"
sys.path.append("../../../")

from datasets import load_dataset
from datasets import DatasetDict
import jittor as jt
from jittor import nn, Module
from jittor.dataset import Dataset
jt.flags.use_cuda = True

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.callbacks.progress_callback import RichCallback
from fastNLP.core.callbacks.callback import Callback
from fastNLP.core.dataloaders.jittor_dataloader.fdl import JittorDataLoader

class TextClassificationDataset(Dataset):
    def __init__(self, dataset):
        super(TextClassificationDataset, self).__init__()
        self.dataset = dataset
        self.set_attrs(total_len=len(dataset))

    def __getitem__(self, idx):
        return {"x": self.dataset["input_ids"][idx], "y": self.dataset["label"][idx]}


class LSTM(Module):
    
    def __init__(self, num_of_words, hidden_size, features):

        self.embedding = nn.Embedding(num_of_words, features)
        self.lstm = nn.LSTM(features, hidden_size, batch_first=True)
        self.layer = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.hidden_size = hidden_size
        self.features = features

    def init_hidden(self, x):
        # batch_first
        batch_size = x.shape[0]
        h0 = jt.randn(1, batch_size, self.hidden_size)
        c0 = jt.randn(1, batch_size, self.hidden_size)

        return h0, c0

    def execute(self, input_ids):

        output = self.embedding(input_ids)
        # TODO 去除padding
        output, (h, c) = self.lstm(output, self.init_hidden(output))
        # len, batch, hidden_size
        output = self.layer(output[-1])

        return output

    def train_step(self, x, y):
        x = self(x)
        outputs = self.loss_fn(x, y)
        return {"loss": outputs}

    def evaluate_step(self, x, y):
        x = self(x)
        return {"pred": x, "target": y.reshape((-1,))}


class PrintWhileTrainingCallBack(Callback):
    """
    通过该Callback实现训练过程中loss的输出
    """

    def __init__(self, print_every_epoch, print_every_batch):
        self.print_every_epoch = print_every_epoch
        self.print_every_batch = print_every_batch

        self.loss = 0
        self.start = 0
        self.epoch_start = 0

    def on_train_begin(self, trainer):
        """
        在训练开始前输出信息
        """
        print("Start training. Total {} epochs and {} batches in each epoch.".format(
            trainer.n_epochs, trainer.num_batches_per_epoch
        ))
        self.start = time.time()

    def on_before_backward(self, trainer, outputs):
        """
        每次反向传播前统计loss，用于计算平均值
        """
        loss = trainer.extract_loss_from_outputs(outputs)
        loss = trainer.driver.tensor_to_numeric(loss)
        self.loss += loss

    def on_train_epoch_begin(self, trainer):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer):
        """
        在每经过一定epoch或最后一个epoch时输出当前epoch的平均loss和使用时间
        """
        if trainer.cur_epoch_idx % self.print_every_epoch == 0 \
            or trainer.cur_epoch_idx == trainer.n_epochs:
            print("Epoch: {} Loss: {} Current epoch training time: {}s".format(
                trainer.cur_epoch_idx, self.loss / trainer.num_batches_per_epoch, time.time() - self.epoch_start
            ))
        # 将loss清零
        self.loss = 0
    
    def on_train_batch_end(self, trainer):
        """
        在每经过一定batch或最后一个batch时输出当前epoch截止目前的平均loss
        """
        if trainer.batch_idx_in_epoch % self.print_every_batch == 0 \
            or trainer.batch_idx_in_epoch == trainer.num_batches_per_epoch:
            print("\tBatch: {} Loss: {}".format(
                trainer.batch_idx_in_epoch, self.loss / trainer.batch_idx_in_epoch
            ))

    def on_train_end(self, trainer):
        print("Total training time: {}s".format(time.time() - self.start))

    
def process_data(ds: DatasetDict, vocabulary: Vocabulary, max_len=256) -> DatasetDict:
    # 分词
    ds = ds.map(lambda x: {"input_ids": text_to_id(vocabulary, x["text"], max_len)})
    ds.set_format(type="numpy", columns=ds.column_names)
    return ds

def set_vocabulary(vocab, dataset):

    for data in dataset:
        vocab.update(data["text"].split())
    return vocab

def text_to_id(vocab, text: str, max_len):
    text = text.split()
    # to index
    ids = [vocab.to_index(word) for word in text]
    # padding
    ids += [vocab.padding_idx] * (max_len - len(text))
    return ids[:max_len]

def get_dataset(name, max_len, train_format="", test_format=""):

    # datasets
    train_dataset = load_dataset(name, split="train" + train_format).shuffle(seed=123)
    test_dataset = load_dataset(name, split="test" + test_format).shuffle(seed=321)
    split = train_dataset.train_test_split(test_size=0.2, seed=123)
    train_dataset = split["train"]
    val_dataset = split["test"]

    vocab = Vocabulary()
    vocab = set_vocabulary(vocab, train_dataset)
    vocab = set_vocabulary(vocab, val_dataset)

    train_dataset = process_data(train_dataset, vocab, max_len)
    val_dataset = process_data(val_dataset, vocab, max_len)
    test_dataset = process_data(test_dataset, vocab, max_len)

    return TextClassificationDataset(train_dataset), TextClassificationDataset(val_dataset), \
            TextClassificationDataset(test_dataset), vocab

if __name__ == "__main__":

    # 训练参数
    max_len = 20
    epochs = 40
    lr = 1
    batch_size = 64

    features = 100
    hidden_size = 128

    # 获取数据集
    # imdb.py SetFit/sst2
    train_data, val_data, test_data, vocab = get_dataset("SetFit/sst2", max_len, "", "")
    # 使用dataloader
    train_dataloader = JittorDataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = JittorDataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_dataloader = JittorDataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=False,
    )

    # 初始化模型
    model = LSTM(len(vocab), hidden_size, features)

    # 优化器
    # 也可以是多个优化器的list
    optimizer = nn.SGD(model.parameters(), lr)

    # Metrics
    metrics = {"acc": Accuracy()}

    # callbacks
    callbacks = [
        PrintWhileTrainingCallBack(print_every_epoch=1, print_every_batch=10),
        # RichCallback(), # print_every参数默认为1，即每一个batch更新一次进度条
    ]

    trainer = Trainer(
        model=model,
        driver="jittor",
        device=[0,1,2,3,4],
        optimizers=optimizer,
        train_dataloader=train_dataloader,
        evaluate_dataloaders=val_dataloader,
        validate_every=-1,
        input_mapping=None,
        output_mapping=None,
        metrics=metrics,
        n_epochs=epochs,
        callbacks=callbacks,
        # progress_bar="raw"
    )
    trainer.run()