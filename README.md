# fastNLP

[![Build Status](https://travis-ci.org/fastnlp/fastNLP.svg?branch=master)](https://travis-ci.org/fastnlp/fastNLP)
[![codecov](https://codecov.io/gh/fastnlp/fastNLP/branch/master/graph/badge.svg)](https://codecov.io/gh/fastnlp/fastNLP)

fastNLP is a modular Natural Language Processing system based on PyTorch, for fast development of NLP tools. It divides the NLP model based on deep learning into different modules. These modules fall into 4 categories: encoder, interaction, aggregation and decoder, while each category contains different implemented modules. Encoder modules encode the input into some abstract representation, interaction modules make the information in the representation interact with each other, aggregation modules aggregate and reduce information, and decoder modules decode the representation into the output. Most current NLP models could be built on these modules, which vastly simplifies the process of developing NLP models. The architecture of fastNLP is as the figure below:

![](https://github.com/fastnlp/fastNLP/raw/master/fastnlp-architecture.jpg)


## Requirements

- numpy>=1.14.2
- torch==0.4.0
- torchvision>=0.1.8


## Resources

- [Documentation](https://github.com/fastnlp/fastNLP)
- [Source Code](https://github.com/fastnlp/fastNLP)


## Example

### Basic Usage

A typical fastNLP routine is composed of four phases: loading dataset, pre-processing data, constructing model and training model.
```python
from fastNLP.core.preprocess import ClassPreprocess
from fastNLP.core.predictor import ClassificationInfer
from fastNLP.core.trainer import ClassificationTrainer
from fastNLP.loader.dataset_loader import ClassDatasetLoader
from fastNLP.models.base_model import BaseModel
from fastNLP.modules import aggregation
from fastNLP.modules import encoder
from fastNLP.modules import decoder
from fastNLP.core.loss import Loss
from fastNLP.core.optimizer import Optimizer


class ClassificationModel(BaseModel):
    """
    Simple text classification model based on CNN.
    """

    def __init__(self, num_classes, vocab_size):
        super(ClassificationModel, self).__init__()

        self.emb = encoder.Embedding(nums=vocab_size, dims=300)
        self.enc = encoder.Conv(
            in_channels=300, out_channels=100, kernel_size=3)
        self.agg = aggregation.MaxPool()
        self.dec = decoder.MLP(size_layer=[100, num_classes])

    def forward(self, x):
        x = self.emb(x)  # [N,L] -> [N,L,C]
        x = self.enc(x)  # [N,L,C_in] -> [N,L,C_out]
        x = self.agg(x)  # [N,L,C] -> [N,C]
        x = self.dec(x)  # [N,C] -> [N, N_class]
        return x


data_dir = 'save/'  # directory to save data and model
train_path = './data_for_tests/text_classify.txt'  # training set file

# load dataset
ds_loader = ClassDatasetLoader(train_path)
data = ds_loader.load()

# pre-process dataset
pre = ClassPreprocess()
train_set, dev_set = pre.run(data, train_dev_split=0.3, pickle_path=data_dir)
n_classes, vocab_size = pre.num_classes, pre.vocab_size

# construct model
model_args = {
    'num_classes': n_classes,
    'vocab_size': vocab_size
}
model = ClassificationModel(num_classes=n_classes, vocab_size=vocab_size)

# construct trainer
train_args = {
    "epochs": 3,
    "batch_size": 16,
    "pickle_path": data_dir,
    "validate": False,
    "save_best_dev": False,
    "model_saved_path": None,
    "use_cuda": True,
    "loss": Loss("cross_entropy"),
    "optimizer": Optimizer("Adam", lr=0.001)
}
trainer = ClassificationTrainer(**train_args)

# start training
trainer.train(model, train_data=train_set, dev_data=dev_set)

# predict using model
data_infer = [x[0] for x in data]
infer = ClassificationInfer(data_dir)
labels_pred = infer.predict(model.cpu(), data_infer)
print(labels_pred)
```


## Installation
Run the following commands to install fastNLP package.
```shell
pip install fastNLP
```

### Cloning From GitHub

If you just want to use fastNLP, use:
```shell
git clone https://github.com/fastnlp/fastNLP
cd fastNLP
```

### PyTorch Installation

Visit the [PyTorch official website] for installation instructions based on your system. In general, you could use:
```shell
# using conda
conda install pytorch torchvision -c pytorch
# or using pip
pip3 install torch torchvision
```


## Project Structure

```
FastNLP
├── docs
├── fastNLP
│   ├── core
│   │   ├── action.py
│   │   ├── __init__.py
│   │   ├── loss.py
│   │   ├── metrics.py
│   │   ├── optimizer.py
│   │   ├── predictor.py
│   │   ├── preprocess.py
│   │   ├── README.md
│   │   ├── tester.py
│   │   └── trainer.py
│   ├── fastnlp.py
│   ├── __init__.py
│   ├── loader
│   │   ├── base_loader.py
│   │   ├── config_loader.py
│   │   ├── dataset_loader.py
│   │   ├── embed_loader.py
│   │   ├── __init__.py
│   │   └── model_loader.py
│   ├── models
│   ├── modules
│   │   ├── aggregation
│   │   ├── decoder
│   │   ├── encoder
│   │   ├── __init__.py
│   │   ├── interaction
│   │   ├── other_modules.py
│   │   └── utils.py
│   └── saver
├── LICENSE
├── README.md
├── reproduction
├── requirements.txt
├── setup.py
└── test
    ├── core
    ├── data_for_tests
    ├── __init__.py
    ├── loader
    ├── modules
    └── readme_example.py

```
