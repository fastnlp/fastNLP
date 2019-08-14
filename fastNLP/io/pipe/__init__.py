

"""
Pipe用于处理数据，所有的Pipe都包含一个process(DataBundle)方法，传入一个DataBundle对象, 在传入DataBundle上进行原位修改，并将其返回；
    process_from_file(paths)传入的文件路径，返回一个DataBundle。process(DataBundle)或者process_from_file(paths)的返回DataBundle
    中的DataSet一般都包含原文与转换为index的输入，以及转换为index的target；除了DataSet之外，还会包含将field转为index时所建立的词表。

"""