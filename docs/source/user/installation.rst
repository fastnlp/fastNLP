===============
安装指南
===============

.. contents::
   :local:

fastNLP 依赖如下包::

    numpy>=1.14.2
    torch>=1.0.0
    tqdm>=4.28.1
    nltk>=3.4.1
    requests
    spacy
    prettytable>=0.7.2

其中torch的安装可能与操作系统及 CUDA 的版本相关，请参见 `PyTorch 官网 <https://pytorch.org/>`_ 。
在依赖包安装完成的情况，您可以在命令行执行如下指令完成安装

..  code:: shell

   >>> pip install fastNLP
   >>> python -m spacy download en
