.. role:: hidden
    :class: hidden-section

fastNLP.modules
===================================

.. contents:: fastNLP.modules
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: fastNLP.modules.torch

Torch Modules
----------------

Encoder
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ConvMaxpool
   LSTM
   Seq2SeqEncoder
   TransformerSeq2SeqEncoder
   LSTMSeq2SeqEncoder
   StarTransformer
   VarRNN
   VarLSTM
   VarGRU

Decoder
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ConditionalRandomField
   State
   Seq2SeqDecoder
   LSTMSeq2SeqDecoder
   TransformerSeq2SeqDecoder
   MLP

.. autosummary::
   :toctree: generated
   :nosignatures:

   allowed_transitions

Generator
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   SequenceGenerator

Dropout
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   TimestepDropout

.. currentmodule:: fastNLP.modules.torch.attention

Attention
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   MultiHeadAttention
   BiAttention
   SelfAttention

.. currentmodule:: fastNLP.modules

Mix Modules
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   torch2paddle
   paddle2torch
   torch2jittor
   jittor2torch
