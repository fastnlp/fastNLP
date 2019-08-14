
"""
Loader用于读取数据，并将内容读取到 :class:`~fastNLP.DataSet` 或者 :class:`~fastNLP.io.DataBundle`中。所有的Loader都支持以下的
    三个方法： __init__()，_load(), loads(). 其中__init__()用于申明读取参数，以及说明该Loader支持的数据格式，读取后Dataset中field
    ; _load(path)方法传入一个文件路径读取单个文件，并返回DataSet; load(paths)用于读取文件夹下的文件，并返回DataBundle, load()方法
    支持以下三种类型的参数

    Example::
        (0) 如果传入None，将尝试自动下载数据集并缓存。但不是所有的数据都可以直接下载。
        (1) 如果传入的是一个文件path，则返回的DataBundle包含一个名为train的DataSet可以通过data_bundle.datasets['train']获取
		(2) 传入的是一个文件夹目录，将读取的是这个文件夹下文件名中包含'train', 'test', 'dev'的文件，其它文件会被忽略。
			假设某个目录下的文件为
				-train.txt
				-dev.txt
				-test.txt
				-other.txt
			Loader().load('/path/to/dir')读取，返回的data_bundle中可以用data_bundle.datasets['train'], data_bundle.datasets['dev'],
			    data_bundle.datasets['test']获取对应的DataSet，其中other.txt的内容会被忽略。
			假设某个目录下的文件为
				-train.txt
				-dev.txt
			Loader().load('/path/to/dir')读取，返回的data_bundle中可以用data_bundle.datasets['train'], data_bundle.datasets['dev']获取
				对应的DataSet。
		(3) 传入一个dict，key为dataset的名称，value是该dataset的文件路径。
			paths = {'train':'/path/to/train', 'dev': '/path/to/dev', 'test':'/path/to/test'}
			Loader().load(paths)  # 返回的data_bundle可以通过以下的方式获取相应的DataSet, data_bundle.datasets['train'], data_bundle.datasets['dev'],
				data_bundle.datasets['test']

"""

