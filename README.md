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
- tensorboardX


## Resources

- [Documentation](https://fastnlp.readthedocs.io/en/latest/)
- [Source Code](https://github.com/fastnlp/fastNLP)

## Installation
Run the following commands to install fastNLP package.
```shell
pip install fastNLP
```


## Project Structure

<table>
<tr>
    <td><b> fastNLP </b></td>
    <td> an open-source NLP library </td>
</tr>
<tr>
    <td><b> fastNLP.core </b></td>
    <td> trainer, tester, predictor </td>
</tr>
<tr>
    <td><b> fastNLP.loader </b></td>
    <td> all kinds of loaders/readers </td>
</tr>
<tr>
    <td><b> fastNLP.models </b></td>
    <td> a collection of NLP models </td>
</tr>
<tr>
    <td><b> fastNLP.modules </b></td>
    <td> a collection of PyTorch sub-models/components/wheels </td>
</tr>
<tr>
    <td><b> fastNLP.saver </b></td>
    <td> all kinds of savers/writers </td>
</tr>
<tr>
    <td><b> fastNLP.fastnlp </b></td>
    <td> a high-level interface for prediction </td>
</tr>
</table>