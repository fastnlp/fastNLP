# fastNLP

[![Build Status](https://travis-ci.org/fastnlp/fastNLP.svg?branch=master)](https://travis-ci.org/fastnlp/fastNLP)
[![codecov](https://codecov.io/gh/fastnlp/fastNLP/branch/master/graph/badge.svg)](https://codecov.io/gh/fastnlp/fastNLP)

fastNLP is a modular Natural Language Processing system based on PyTorch, for fast development of NLP tools. It divides the NLP model based on deep learning into different modules. These modules fall into 4 categories: encoder, interaction, aggregation and decoder, while each category contains different implemented modules. Encoder modules encode the input into some abstract representation, interaction modules make the information in the representation interact with each other, aggregation modules aggregate and reduce information, and decoder modules decode the representation into the output. Most current NLP models could be built on these modules, which vastly simplifies the process of developing NLP models. The architecture of fastNLP is as the figure below:

![](fastnlp-architecture.pdf)


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
from fastNLP.models.base_model import BaseModel
from fastNLP.modules import encoder
from fastNLP.modules import aggregation

from fastNLP.loader.dataset_loader import ClassDatasetLoader
from fastNLP.loader.preprocess import ClassPreprocess
from fastNLP.core.trainer import ClassificationTrainer
from fastNLP.core.inference import ClassificationInfer


class ClassificationModel(BaseModel):
    """
    Simple text classification model based on CNN.
    """

    def __init__(self, class_num, vocab_size):
        super(ClassificationModel, self).__init__()

        self.embed = encoder.Embedding(nums=vocab_size, dims=300)
        self.conv = encoder.Conv(
            in_channels=300, out_channels=100, kernel_size=3)
        self.pool = aggregation.MaxPool()
        self.output = encoder.Linear(input_size=100, output_size=class_num)

    def forward(self, x):
        x = self.embed(x)  # [N,L] -> [N,L,C]
        x = self.conv(x)  # [N,L,C_in] -> [N,L,C_out]
        x = self.pool(x)  # [N,L,C] -> [N,C]
        x = self.output(x)  # [N,C] -> [N, N_class]
        return x


data_dir = 'data'  # directory to save data and model
train_path = 'test/data_for_tests/text_classify.txt'  # training set file

# load dataset
ds_loader = ClassDatasetLoader("train", train_path)
data = ds_loader.load()

# pre-process dataset
pre = ClassPreprocess(data_dir)
vocab_size, n_classes = pre.process(data, "data_train.pkl")

# construct model
model_args = {
    'num_classes': n_classes,
    'vocab_size': vocab_size
}
model = ClassificationModel(class_num=n_classes, vocab_size=vocab_size)

# train model
train_args = {
    "epochs": 20,
    "batch_size": 50,
    "pickle_path": data_dir,
    "validate": False,
    "save_best_dev": False,
    "model_saved_path": None,
    "use_cuda": True,
    "learn_rate": 1e-3,
    "momentum": 0.9}
trainer = ClassificationTrainer(train_args)
trainer.train(model)

# predict using model
seqs = [x[0] for x in data]
infer = ClassificationInfer(data_dir)
labels_pred = infer.predict(model, seqs)
```


## Installation

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
│   └── quick_tutorial.md
├── fastNLP
│   ├── action
│   │   ├── action.py
│   │   ├── inference.py
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── optimizer.py
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
│   │   ├── model_loader.py
│   │   └── preprocess.py
│   ├── models
│   │   ├── base_model.py
│   │   ├── char_language_model.py
│   │   ├── cnn_text_classification.py
│   │   ├── __init__.py
│   │   └── sequence_modeling.py
│   ├── modules
│   │   ├── aggregation
│   │   │   ├── attention.py
│   │   │   ├── avg_pool.py
│   │   │   ├── __init__.py
│   │   │   ├── kmax_pool.py
│   │   │   ├── max_pool.py
│   │   │   └── self_attention.py
│   │   ├── decoder
│   │   │   ├── CRF.py
│   │   │   └── __init__.py
│   │   ├── encoder
│   │   │   ├── char_embedding.py
│   │   │   ├── conv_maxpool.py
│   │   │   ├── conv.py
│   │   │   ├── embedding.py
│   │   │   ├── __init__.py
│   │   │   ├── linear.py
│   │   │   ├── lstm.py
│   │   │   ├── masked_rnn.py
│   │   │   └── variational_rnn.py
│   │   ├── __init__.py
│   │   ├── interaction
│   │   │   └── __init__.py
│   │   ├── other_modules.py
│   │   └── utils.py
│   └── saver
│       ├── base_saver.py
│       ├── __init__.py
│       ├── logger.py
│       └── model_saver.py
├── LICENSE
├── README.md
├── reproduction
│   ├── Char-aware_NLM
│   │  
│   ├── CNN-sentence_classification
│   │  
│   ├── HAN-document_classification
│   │  
│   └── LSTM+self_attention_sentiment_analysis
|
├── requirements.txt
├── setup.py
└── test
    ├── data_for_tests
    │   ├── charlm.txt
    │   ├── config
    │   ├── cws_test
    │   ├── cws_train
    │   ├── people_infer.txt
    │   └── people.txt
    ├── test_charlm.py
    ├── test_cws.py
    ├── test_fastNLP.py
    ├── test_loader.py
    ├── test_seq_labeling.py
    ├── test_tester.py
    └── test_trainer.py
```
