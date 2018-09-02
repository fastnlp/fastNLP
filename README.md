# fastNLP

[![Build Status](https://travis-ci.org/fastnlp/fastNLP.svg?branch=master)](https://travis-ci.org/fastnlp/fastNLP)
[![codecov](https://codecov.io/gh/fastnlp/fastNLP/branch/master/graph/badge.svg)](https://codecov.io/gh/fastnlp/fastNLP)
[![PyPI version](https://badge.fury.io/py/fastNLP.svg)](https://badge.fury.io/py/fastNLP)
![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)
[![Documentation Status](https://readthedocs.org/projects/fastnlp/badge/?version=latest)](http://fastnlp.readthedocs.io/?badge=latest)

fastNLP is a modular Natural Language Processing system based on PyTorch, for fast development of NLP tools. It divides the NLP model based on deep learning into different modules. These modules fall into 4 categories: encoder, interaction, aggregation and decoder, while each category contains different implemented modules. Encoder modules encode the input into some abstract representation, interaction modules make the information in the representation interact with each other, aggregation modules aggregate and reduce information, and decoder modules decode the representation into the output. Most current NLP models could be built on these modules, which vastly simplifies the process of developing NLP models. The architecture of fastNLP is as the figure below:

![](https://github.com/fastnlp/fastNLP/raw/master/fastnlp-architecture.jpg)


## Requirements

- numpy>=1.14.2
- torch==0.4.0
- torchvision>=0.1.8


## Resources

- [Documentation](https://fastnlp.readthedocs.io/en/latest/)
- [Source Code](https://github.com/fastnlp/fastNLP)



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
