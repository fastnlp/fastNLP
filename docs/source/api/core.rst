.. role:: hidden
   :class: hidden-section

fastNLP.core
===================================

.. contents:: fastNLP.core
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: fastNLP.core

Callbacks
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Callback
   CheckpointCallback
   ProgressCallback
   RichCallback
   TqdmCallback
   RawTextCallback
   LRSchedCallback
   LoadBestModelCallback
   EarlyStopCallback
   MoreEvaluateCallback
   ResultsMonitor
   HasMonitorCallback
   FitlogCallback
   TimerCallback

   ExtraInfoStatistics

Torch Callbacks
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   TorchWarmupCallback
   TorchGradClipCallback

Callback Event
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Event
   Filter

Collators
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Collator
   Padder
   NullPadder
   RawNumberPadder
   RawSequencePadder

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_padded_numpy_array

Numpy Padder
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   NumpyNumberPadder
   NumpySequencePadder
   NumpyTensorPadder

Torch Padder
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   TorchNumberPadder
   TorchSequencePadder
   TorchTensorPadder

Paddle Padder
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   PaddleNumberPadder
   PaddleTensorPadder
   PaddleSequencePadder

Oneflow Padder
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   OneflowNumberPadder
   OneflowTensorPadder
   OneflowSequencePadder

Jittor Padder
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   JittorNumberPadder
   JittorTensorPadder
   JittorSequencePadder

Controllers
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Trainer
   Evaluator

Loop
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Loop
   EvaluateBatchLoop
   TrainBatchLoop

Dataset
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   DataSet
   FieldArray
   Instance

DataLoaders
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   prepare_dataloader

Torch DataLoader
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   TorchDataLoader
   MixDataLoader

.. autosummary::
   :toctree: generated
   :nosignatures:

   prepare_torch_dataloader

Paddle DataLoader
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   PaddleDataLoader

.. autosummary::
   :toctree: generated
   :nosignatures:

   prepare_paddle_dataloader

Jittor DataLoader
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   JittorDataLoader

.. autosummary::
   :toctree: generated
   :nosignatures:

   prepare_jittor_dataloader

Oneflow DataLoader
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   OneflowDataLoader

.. autosummary::
   :toctree: generated
   :nosignatures:

   prepare_oneflow_dataloader

Drivers
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Driver

Torch Driver
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   TorchDriver
   TorchSingleDriver
   TorchDDPDriver
   DeepSpeedDriver
   FairScaleDriver
   TorchFSDPDriver

Paddle Driver
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   PaddleDriver
   PaddleSingleDriver
   PaddleFleetDriver

Jittor Driver
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   JittorDriver
   JittorSingleDriver
   JittorMPIDriver

Oneflow Driver
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   OneflowDriver
   OneflowSingleDriver
   OneflowDDPDriver

Utils
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   torch_seed_everything
   paddle_seed_everything
   oneflow_seed_everything
   torch_move_data_to_device
   paddle_move_data_to_device
   oneflow_move_data_to_device

Vocabulary
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Vocabulary

Logger
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   logger
   print

Metrics
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Metric
   Accuracy
   TransformersAccuracy
   SpanFPreRecMetric
   ClassifyFPreRecMetric
   BLEU
   ROUGE
   Perplexity

Samplers
----------------

ReproducibleBatchSampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ReproduceBatchSampler
   BucketedBatchSampler
   ReproducibleBatchSampler
   RandomBatchSampler

ReproducibleSampler
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ReproducibleSampler
   RandomSampler
   SequentialSampler
   SortedSampler

UnrepeatedSampler
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   UnrepeatedSampler
   UnrepeatedRandomSampler
   UnrepeatedSortedSampler
   UnrepeatedSequentialSampler

MixSampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   MixSampler
   DopedSampler
   MixSequentialSampler
   PollingSampler

Utils
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   cache_results
   f_rich_progress
   auto_param_call
   f_tqdm_progress
   seq_len_to_mask
