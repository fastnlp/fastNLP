Distributed Parallel Training
=============================

原理
----

随着深度学习模型越来越复杂，单个GPU可能已经无法满足正常的训练。比如BERT等预训练模型，更是在多个GPU上训练得到的。为了使用多GPU训练，Pytorch框架已经提供了
`nn.DataParallel <https://pytorch.org/docs/stable/nn.html#dataparallel>`_ 以及
`nn.DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#distributeddataparallel>`_ 两种方式的支持。
`nn.DataParallel <https://pytorch.org/docs/stable/nn.html#dataparallel>`_
很容易使用，但是却有着GPU负载不均衡，单进程速度慢等缺点，无法发挥出多GPU的全部性能。因此，分布式的多GPU训练方式
`nn.DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#distributeddataparallel>`_
是更好的选择。然而，因为分布式训练的特点，
`nn.DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#distributeddataparallel>`_
常常难以理解和使用，也很难debug。所以，在使用分布式训练之前，需要理解它的原理。

在使用
`nn.DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#distributeddataparallel>`_
时，模型会被复制到所有使用的GPU，通常每个GPU上存有一个模型，并被一个单独的进程控制。这样有N块GPU，就会产生N个进程。当训练一个batch时，这一batch会被分为N份，每个进程会使用batch的一部分进行训练，然后在必要时进行同步，并通过网络传输需要同步的数据。这时，只有模型的梯度会被同步，而模型的参数不会，所以能缓解大部分的网络传输压力，网络传输不再是训练速度的瓶颈之一。你可能会好奇，不同步模型的参数，怎么保证不同进程所训练的模型相同？只要每个进程初始的模型是同一个，具有相同的参数，而之后每次更新，都使用相同的梯度，就能保证梯度更新后的模型也具有相同的参数了。

为了让每个进程的模型初始化完全相同，通常这N个进程都是由单个进程复制而来的，这时需要对分布式的进程进行初始化，建立相互通信的机制。在
Pytorch 中，我们用
`distributed.init_process_group <https://pytorch.org/docs/stable/distributed.html#initialization>`_
函数来完成，需要在程序开头就加入这一步骤。初始化完成后，每一个进程用唯一的编号
``rank`` 进行区分，从 0 到 N-1递增，一般地，我们将 ``rank`` 为 0
的进程当作主进程，而其他 ``rank`` 的进程为子进程。每个进程还要知道
``world_size`` ，即分布式训练的总进程数
N。训练时，每个进程使用batch的一部分，互相不能重复，这里通过
`nn.utils.data.DistributedSampler <https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html>`_
来实现。

使用方式
--------

Pytorch的分布式训练使用起来非常麻烦，难以理解，可以从给出的\ `官方教程 <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_ \ 中看到。而\ ``fastNLP``
提供了
``DistTrainer``\ ，将大部分的分布式训练的细节进行了封装，只需简单的改动训练代码，就能直接用上分布式训练。那么，具体怎么将普通的训练代码改成支持分布式训练的代码呢。下面我们来讲一讲分布式训练的完整流程。通常，分布式程序的多个进程是单个进程的复制。假设我们用N个GPU进行分布式训练，我们需要启动N个进程，这时，在命令行使用：

.. code:: shell

   python -m torch.distributed.launch --nproc_per_node=N train_script.py --args

其中\ ``N``\ 是需要启动的进程数，\ ``train_script.py``\ 为训练代码，\ ``--args``\ 是自定义的命令行参数。在启动了N个进程之后，如果我们在\ ``train_script.py``\ 的训练代码中正常配置，分布式训练就能正常进行。

此外，还可以使用环境变量\ ``CUDA_VISIBLE_DEVICES``\ 设置指定的GPU，比如在8卡机器上使用编号为4,5,6,7的4块GPU：

.. code:: shell

   CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=N train_script.py --args

在 ``train_script.py``
训练代码中，有一些必须的配置。为了清晰的叙述，这里放一个简单的分布式训练代码，省去多余细节：

.. code:: python

   import torch.distributed as dist
   from fastNLP import DistTrainer, get_local_rank
   import fastNLP as fnlp

   def main(options):
       # options为训练所需的参数，batch_size等
       
       set_seed(options.seed)
       
       # 初始化分布式进程
       dist.init_process_group('nccl')

       ######## 读取数据
       if get_local_rank() != 0:
           dist.barrier() # 先让主进程(rank==0)先执行，进行数据处理，预训模型参数下载等操作，然后保存cache
       data = get_processed_data()
       model = get_model(data.get_vocab("words"), data.get_vocab("target"))
       if get_local_rank() == 0:
           dist.barrier() # 主进程执行完后，其余进程开始读取cache
       ######## 

       # 初始化Trainer，训练等，与普通训练差别不大
       def get_trainer(model, data):
           # 注意设置的callback有两种，一种只在主进程执行，一种在所有进程都执行
           callbacks_master = [fnlp.FitlogCallback()] 
           callbacks_all = [fnlp.WarmupCallback(warmup=options.warmup)]
           trainer = DistTrainer(
               save_path='save',
               train_data=data.get_dataset("train"),
               dev_data=data.get_dataset("dev"),
               model=model,
               loss=fnlp.CrossEntropyLoss(),
               metrics=fnlp.AccuracyMetric(),
               metric_key="acc",
               optimizer=fnlp.AdamW(model.parameters(), lr=options.lr),
               callbacks_master=callbacks_master, # 仅在主进程执行（如模型保存，日志记录）
               callbacks_all=callbacks_all,    # 在所有进程都执行（如梯度裁剪，学习率衰减）
               batch_size_per_gpu=options.batch_size, # 指定每个GPU的batch大小
               update_every=options.update,
               n_epochs=options.epochs,
               use_tqdm=True,
           )
           return trainer
       
       trainer = get_trainer(model, data)
       trainer.train()

指定进程编号
^^^^^^^^^^^^

首先，为了区分不同的进程，初始时需要对每个进程传入\ ``rank``\ 。这里一般分为\ ``node_rank``\ 和\ ``local_rank``\ ，分别表示进程处于哪一机器以及同机器上处于第几进程。如果在单一机器上，\ ``node_rank``\ 可以省略。\ ``local_rank``\ 一般通过命令行参数\ ``--local_rank``\ 传入，为\ ``int``\ 类型。也可以通过环境变量传入\ ``local_rank``\ ，只需在\ ``torch.distributed.launch``\ 时，使用\ ``--use_env``\ 参数。无论哪种方式，在训练脚本中，都要获取到\ ``local_rank``\ ，用于初始化分布式通信，以及区分进程。如果你使用\ ``fastNLP``\ ，可以通过\ ``fastNLP.get_local_rank``\ 来得到\ ``local_rank``\ 。

初始化进程
^^^^^^^^^^

在获取了\ ``local_rank``\ 等重要参数后，在开始训练前，我们需要建立不同进程的通信和同步机制。这时我们使用\ `torch.distributed.init_process_group <https://pytorch.org/docs/stable/distributed.html#initialization>`_
来完成。通常，我们只需要 ``torch.distributed.init_process_group('nccl')``
来指定使用\ ``nccl``\ 后端来进行同步即可。其他参数程序将读取环境变量自动设置。如果想手动设置这些参数，比如，使用TCP进行通信，可以设置：

.. code:: python

   init_process_group('nccl', init_method='tcp://localhost:55678',
                     rank=args.rank, world_size=N)

或者使用文件进行通信：

.. code:: python

   init_process_group('nccl', init_method='file:///mnt/nfs/sharedfile',
                     world_size=N, rank=args.rank)

注意，此时必须显式指定\ ``world_size``\ 和\ ``rank``\ ，具体可以参考
`torch.distributed.init_process_group <https://pytorch.org/docs/stable/distributed.html#initialization>`_
的使用文档。

在初始化分布式通信后，再初始化\ ``DistTrainer``\ ，传入数据和模型，就完成了分布式训练的代码。代码修改完成后，使用上面给出的命令行启动脚本，就能成功运行分布式训练。但是，如果数据处理，训练中的自定义操作比较复杂，则可能需要额外的代码修改。下面列出一些需要特别注意的地方，在使用分布式训练前，请仔细检查这些事项。

注意事项
--------

在执行完
`torch.distributed.init_process_group <https://pytorch.org/docs/stable/distributed.html#initialization>`_
后，我们就可以在不同进程间完成传输数据，进行同步等操作。这些操作都可以在\ `torch.distributed <https://pytorch.org/docs/stable/distributed.html#>`_
中找到。其中，最重要的是
`barrier <https://pytorch.org/docs/stable/distributed.html#torch.distributed.barrier>`_
以及
`get_rank <https://pytorch.org/docs/stable/distributed.html#torch.distributed.get_rank>`_
操作。对于训练而言，我们关心的是读入数据，记录日志，模型初始化，模型参数更新，模型保存等操作。这些操作大多是读写操作，在多进程状态下，这些操作都必须小心进行，否则可能出现难以预料的bug。而在\ ``fastNLP``\ 中，大部分操作都封装在
``DistTrainer`` 中，只需保证数据读入和模型初始化正确即可完成训练。

写操作
^^^^^^

一般而言，读入操作需要在每一个进程都执行，因为每个进程都要使用读入的数据和模型参数进行训练。而写出操作只需在其中一个进程（通常为主进程）执行，因为每一个进程保存的模型都相同，都处于同一训练状态。所以，通常单进程的训练脚本中，只需要修改写出操作的部分，通过加入对进程\ ``rank``\ 的判断，仅让其中一个进程执行写操作：

.. code:: python

   import torch.distributed as dist

   # 仅在主进程才执行
   if dist.get_rank() == 0:
       do_wirte_op()  # 一些写操作 
   dist.barrier()  # 确保写完成后，所有进程再执行（若进程无需读入写出的数据，可以省去）

若使用\ ``fastNLP``\ 中的\ ``DistTrainer``\ ，也可以这样写：

.. code:: python

   # 判断是否是主进程的trainer
   if trainer.is_master:
       do_wirte_op()
   dist.barrier()

读操作
^^^^^^

然而有些时候，我们需要其中一个进程先执行某些操作，等这一进程执行完后，其它进程再执行这一操作。比如，在读入数据时，我们有时需要从网上下载，再处理，将处理好的数据保存，供反复使用。这时，我们不需要所有进程都去下载和处理数据，只需要主进程进行这些操作，其它进程等待。直到处理好的数据被保存后，其他进程再从保存位置直接读入数据。这里可以参考范例代码中的读取数据：

.. code:: python

   if dist.get_rank() != 0:
       dist.barrier()  # 先让主进程(rank==0)先执行，进行数据处理，预训模型参数下载等操作，然后保存cache

   # 这里会自动处理数据，或直接读取保存的cache
   data = get_processed_data()
   model = get_model(data.get_vocab("words"), data.get_vocab("target"))

   if dist.get_rank() == 0:
       dist.barrier()  # 主进程执行完后，其余进程开始读取cache

也可以显式的将主进程和其它进程的操作分开：

.. code:: python

   if dist.get_rank() == 0:
       data = do_data_processing()  # 数据处理
       dist.barrier()
   else:
       dist.barrier()
       data = load_processed_data()  # 读取cache

日志操作
^^^^^^^^

通常，我们需要知道训练的状态，如当前在第几个epoch，模型当前的loss等等。单进程训练时，我们可以直接使用\ ``print``\ 将这些信息输出到命令行或日志文件。然而，在多进程时，\ ``print``\ 会导致同样的信息在每一进程都输出，造成问题。这一问题和写操作类似，也可以通过判断进程的编号之后再输出。问题是，日志通常在训练的很多地方都有输出，逐一加上判断代码是非常繁琐的。这里，建议统一修改为：

.. code:: python

   from fastNLP import logger
   logger.info('....')  # 替换print

在\ ``DistTrainer``\ 中，主进程的\ ``logger``\ 级别为\ ``INFO``\ ，而其它进程为\ ``WARNING``\ 。这样级别为\ ``INFO``\ 的信息只会在主进程输出，不会造成日志重复问题。若需要其它进程中的信息，可以使用\ ``logger.warning``\ 。

注意，\ ``logger``\ 的级别设置只有初始化了\ ``DistTrainer``\ 后才能生效。如果想要在初始化进程后就生效，需要在分布式通信初始化后，执行\ ``init_logger_dist``\ 。

Callback
^^^^^^^^

``fastNLP``\ 的一个特色是可以使用\ ``Callback``\ 在训练时完成各种自定义操作。而这一特色在\ ``DistTrainer``\ 中得以保留。但是，这时需要特别注意\ ``Callback``\ 是否只需要在主进程执行。一些\ ``Callback``\ ，比如调整学习率，梯度裁剪等，会改变模型的状态，因此需要在所有进程上都执行，将它们通过\ ``callback_all``\ 参数传入\ ``DistTrainer``\ 。而另一些\ ``Callback``\ ，比如\ ``fitlog``\ ，保存模型，不会改变模型的状态，而是进行数据写操作，因此仅在主进程上执行，将它们通过\ ``callback_master``\ 传入。

在自定义\ ``Callback``\ 时，请遵循一个原则，改变训练或模型状态的操作在所有进程中执行，而数据写到硬盘请在主进程单独进行。这样就能避免进程间失去同步，或者磁盘写操作的冲突。

Debug
^^^^^

多进程的程序很难进行debug，如果出现问题，可以先参考报错信息进行处理。也可以在程序中多输出日志，定位问题。具体情况，具体分析。在debug时，要多考虑进程同步和异步的操作，判断问题是程序本身导致的，还是由进程间没有同步而产生。

其中，有一个常见问题是程序卡住不动。具体表现为训练暂停，程序没有输出，但是GPU利用率保持100%。这一问题是由进程失去同步导致的。这时只能手动\ ``kill``\ GPU上残留的进程，再检查代码。需要检查进程同步的位置，比如模型\ ``backward()``\ 时，\ ``barrier()``\ 时等。同时，也要检查主进程与其它进程操作不同的位置，比如存储模型，evaluate模型时等。注意，失去同步的位置可能并不是程序卡住的位置，所以需要细致的检查。
