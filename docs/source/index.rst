fastNLP 中文文档
=====================

`fastNLP <https://github.com/fastnlp/fastNLP>`_ 是一款轻量级的自然语言处理（NLP）工具包。你既可以用它来快速地完成一个 NLP 任务，也可以用它在研究中快速构建更复杂的模型。

fastNLP 具有如下的特性：

- 简化训练过程，代码编写方便快捷；

   .. figure:: figures/fastnlp-compare.gif
      :align: center

- 支持 **PaddlePaddle**、**Jittor**、**oneflow** 国产深度学习框架，并且可以方便地在不同框架之间切换；

   .. figure:: figures/torch2paddle.gif
      :align: center

- 支持分布式训练、deepspeed 等多卡训练机制，并且与单卡训练相比代码修改非常少，实现近似的一键切换训练模式。

   .. figure:: figures/ddp1.gif
      :align: center

      分布式训练 - 修改参数


   .. figure:: figures/ddp2.gif
      :align: center

      分布式训练 - 命令行启动


.. toctree::
   :maxdepth: 2
   :caption: 快速上手

   tutorials/basic/index
   tutorials/torch/index
   tutorials/paddle/index


.. toctree::
   :maxdepth: 2
   :caption: API 文档

   fastNLP.core <api/core>
   fastNLP.io <api/io>
   fastNLP.embeddings <api/embeddings>
   fastNLP.models <api/models>
   fastNLP.modules <api/modules>
   fastNLP.envs <api/envs>
   fastNLP.transformers <api/transformers>

.. toctree::
   :maxdepth: 1
   :caption: 说明

   instructions/contribution


索引与搜索
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
