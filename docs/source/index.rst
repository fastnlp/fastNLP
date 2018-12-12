fastNLP documentation
=====================
A Modularized and Extensible Toolkit for Natural Language Processing. Currently still in incubation. 


Introduction
------------

FastNLP is a modular Natural Language Processing system based on
PyTorch, built for fast development of NLP models.

A deep learning NLP model is the composition of three types of modules:

+-----------------------+-----------------------+-----------------------+
| module type           | functionality         | example               |
+=======================+=======================+=======================+
| encoder               | encode the input into | embedding, RNN, CNN,  |
|                       | some abstract         | transformer           |
|                       | representation        |                       |
+-----------------------+-----------------------+-----------------------+
| aggregator            | aggregate and reduce  | self-attention,       |
|                       | information           | max-pooling           |
+-----------------------+-----------------------+-----------------------+
| decoder               | decode the            | MLP, CRF              |
|                       | representation into   |                       |
|                       | the output            |                       |
+-----------------------+-----------------------+-----------------------+


For example:

.. image:: figures/text_classification.png




User's Guide
------------
.. toctree::
   :maxdepth: 2

   user/installation
   user/quickstart


API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
   :maxdepth: 2
   
   fastNLP API <fastNLP>




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
