# TODO 应该保证在__init__.py中的第一时间被调用。


# FASTNLP_DISTRIBUTED_CHECK 用于这样的使用场景：用户可能在同一个脚本中连续使用两个独立的 trainer 实例，然后这两个 trainer 使用的都是 TorchDDPDriver；
#  因为我们在训练完成后不会主动地去关闭 ddp 的通信进程（目前没有），因此第二个 trainer 的 TorchDDPDriver 不会去真正地初始化 ddp 环境，而是会沿用
#  第一个 trainer 的 TorchDDPDriver 所开启的 ddp 环境；
# 但是注意当第二个 TorchDDPDriver 的机器数量和使用的显卡数量（对应进程数量）发生变化时，这一沿用会造成严重的错误；因此在 TorchDDPDriver 第一次启动 ddp
#  环境后，我们会将 FASTNLP_DISTRIBUTED_CHECK 注入到环境变量中；从而在第二个 TorchDDPDriver 启动的时候会检测到该值，然后去检验当前使用的机器数量和每个机器
#  上的进程的数量是否相等；
FASTNLP_DISTRIBUTED_CHECK = "FASTNLP_DISTRIBUTED_CHECK"

# 每一个 分布式的 driver 都应当正确地设立该值；
# FASTNLP_GLOBAL_RANK 用于给 fastNLP.core.utils.distributed.rank_zero_call 进行正确的配置。这是因为 TorchDDPDriver 初始化 ddp 环境的
#  方式是开启多个和主进程基本一样的子进程，然后将所有代码从前到后完整地运行一遍。而在运行到 TorchDDPDriver 中的设立一些变量的正确地值之前，就已经
#  运行到了某些需要区分主进程和其它进程的代码。
# 因为考虑到用户可能在 Trainer 实例化前调用该函数修饰器，因此我们需要通过环境变量的方式在每一个子进程开始开始运行到被修饰的函数前就将
#  rank_zero_call 的 rank 值设立正确；
FASTNLP_GLOBAL_RANK = "FASTNLP_GLOBAL_RANK"

# FASTNLP_LOG_LEVEL 的使用场景和 FASTNLP_GLOBAL_RANK 类似，即用户在使用我们 log 的时候是在 trainer.run 之前的，这时我们要提前通过
#  环境变量将该值设立正确；
FASTNLP_LOG_LEVEL = "FASTNLP_LOG_LEVEL"


# todo 每一个分布式的 driver 都应当正确地设立该值；具体可见 ddp；
# FASTNLP_LAUNCH_TIME 记录了当前 fastNLP 脚本启动的时间。
FASTNLP_LAUNCH_TIME = "FASTNLP_LAUNCH_TIME"


# FASTNLP_GLOBAL_SEED 用于每个子进程随机数种子的正确设置；
FASTNLP_GLOBAL_SEED = "FASTNLP_GLOBAL_SEED"

# FASTNLP_SEED_WORKERS 用于 pytorch dataloader work_init_fn 的正确的设置；
FASTNLP_SEED_WORKERS = "FASTNLP_SEED_WORKERS"

# 用于设置 fastNLP 使用的 backend 框架
FASTNLP_BACKEND = 'FASTNLP_BACKEND'

# 用于保存用户传入的 CUDA_VISIBLE_DEVICES，目前在paddle中有使用，用户不需要使用
USER_CUDA_VISIBLE_DEVICES = 'USER_CUDA_VISIBLE_DEVICES'

# 用于在 torch.distributed.launch 时移除传入的 rank ，在 pytorch 中有使用。值的可选为 [0, 1]
FASTNLP_REMOVE_LOCAL_RANK = 'FASTNLP_REMOVE_LOCAL_RANK'

# todo 注释
FASTNLP_BACKEND_LAUNCH = "FASTNLP_BACKEND_LAUNCH"


# todo 注释 直接使用的变量
FASTNLP_MODEL_FILENAME = "fastnlp_model.pkl.tar"
FASTNLP_CHECKPOINT_FILENAME = "fastnlp_checkpoint.pkl.tar"

