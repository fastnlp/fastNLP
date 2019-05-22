class Const:
    """
    fastNLP中field命名常量。
    
    .. todo::
        把下面这段改成表格
        
    具体列表::

        INPUT       模型的序列输入      words（复数words1, words2）
        CHAR_INPUT  模型character输入  chars（复数chars1， chars2）
        INPUT_LEN   序列长度           seq_len（复数seq_len1，seq_len2）
        OUTPUT      模型输出           pred（复数pred1， pred2）
        TARGET      真实目标           target（复数target1，target2）
        LOSS        损失函数           loss （复数loss1，loss2）

    """
    INPUT = 'words'
    CHAR_INPUT = 'chars'
    INPUT_LEN = 'seq_len'
    OUTPUT = 'pred'
    TARGET = 'target'
    LOSS = 'loss'

    @staticmethod
    def INPUTS(i):
        """得到第 i 个 ``INPUT`` 的命名"""
        i = int(i) + 1
        return Const.INPUT + str(i)

    @staticmethod
    def CHAR_INPUTS(i):
        """得到第 i 个 ``CHAR_INPUT`` 的命名"""
        i = int(i) + 1
        return Const.CHAR_INPUT + str(i)

    @staticmethod
    def INPUT_LENS(i):
        """得到第 i 个 ``INPUT_LEN`` 的命名"""
        i = int(i) + 1
        return Const.INPUT_LEN + str(i)

    @staticmethod
    def OUTPUTS(i):
        """得到第 i 个 ``OUTPUT`` 的命名"""
        i = int(i) + 1
        return Const.OUTPUT + str(i)

    @staticmethod
    def TARGETS(i):
        """得到第 i 个 ``TARGET`` 的命名"""
        i = int(i) + 1
        return Const.TARGET + str(i)

    @staticmethod
    def LOSSES(i):
        """得到第 i 个 ``LOSS`` 的命名"""
        i = int(i) + 1
        return Const.LOSS + str(i)
