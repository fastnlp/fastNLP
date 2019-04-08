# fastNLP

[![Build Status](https://travis-ci.org/fastnlp/fastNLP.svg?branch=master)](https://travis-ci.org/fastnlp/fastNLP)
[![codecov](https://codecov.io/gh/fastnlp/fastNLP/branch/master/graph/badge.svg)](https://codecov.io/gh/fastnlp/fastNLP)
[![PyPI version](https://badge.fury.io/py/fastNLP.svg)](https://badge.fury.io/py/fastNLP)
![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)
[![Documentation Status](https://readthedocs.org/projects/fastnlp/badge/?version=latest)](http://fastnlp.readthedocs.io/?badge=latest)

*Very sad day for all of us.  @FengZiYjun  is no more with us.  May his soul rest in peace. We will miss you very very much!*

FastNLP is a modular Natural Language Processing system based on PyTorch, built for fast development of NLP models. 

A deep learning NLP model is the composition of three types of modules:
<table>
<tr>
    <td><b> module type </b></td>
    <td><b> functionality </b></td>
    <td><b> example </b></td>
</tr>
<tr>
    <td> encoder </td>
    <td> encode the input into some abstract representation </td>
    <td> embedding, RNN, CNN, transformer
</tr>
<tr>
    <td> aggregator </td>
    <td> aggregate and reduce information </td>
    <td> self-attention, max-pooling </td>
</tr>
<tr>
    <td> decoder </td>
    <td> decode the representation into the output </td>
    <td> MLP, CRF </td>
</tr>
</table>

For example:

![](docs/source/figures/text_classification.png)

## Requirements

- Python>=3.6
- numpy>=1.14.2
- torch>=0.4.0
- tensorboardX
- tqdm>=4.28.1


## Resources

- [Tutorials](https://github.com/fastnlp/fastNLP/tree/master/tutorials)
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
    <td><b> fastNLP.api </b></td>
    <td> APIs for end-to-end prediction </td>
</tr>
<tr>
    <td><b> fastNLP.core </b></td>
    <td> data representation & train/test procedure </td>
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
    <td><b> fastNLP.io </b></td>
    <td> readers & savers </td>
</tr>
</table>
