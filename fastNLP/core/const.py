r"""
fastNLP包当中的field命名均符合一定的规范，该规范由fastNLP.Const类进行定义。
"""

__all__ = [
    "Const"
]


class Const:
    r"""
    fastNLP中field命名常量。
    
    .. todo::
        把下面这段改成表格
        
    具体列表::

        INPUT       模型的序列输入      words（具有多列words时，依次使用words1, words2, ）
        CHAR_INPUT  模型character输入  chars（具有多列chars时，依次使用chars1， chars2）
        INPUT_LEN   序列长度           seq_len（具有多列seq_len时，依次使用seq_len1，seq_len2）
        OUTPUT      模型输出           pred（具有多列pred时，依次使用pred1， pred2）
        TARGET      真实目标           target（具有多列target时，依次使用target1，target2）
        LOSS        损失函数           loss （具有多列loss时，依次使用loss1，loss2）
        RAW_WORD    原文的词           raw_words  (具有多列raw_words时，依次使用raw_words1, raw_words2)
        RAW_CHAR    原文的字           raw_chars  (具有多列raw_chars时，依次使用raw_chars1, raw_chars2)

    """
    INPUT = 'words'
    CHAR_INPUT = 'chars'
    INPUT_LEN = 'seq_len'
    OUTPUT = 'pred'
    TARGET = 'target'
    LOSS = 'loss'
    RAW_WORD = 'raw_words'
    RAW_CHAR = 'raw_chars'
    
    @staticmethod
    def INPUTS(i):
        r"""得到第 i 个 ``INPUT`` 的命名"""
        i = int(i) + 1
        return Const.INPUT + str(i)
    
    @staticmethod
    def CHAR_INPUTS(i):
        r"""得到第 i 个 ``CHAR_INPUT`` 的命名"""
        i = int(i) + 1
        return Const.CHAR_INPUT + str(i)
    
    @staticmethod
    def RAW_WORDS(i):
        r"""得到第 i 个 ``RAW_WORDS`` 的命名"""
        i = int(i) + 1
        return Const.RAW_WORD + str(i)
    
    @staticmethod
    def RAW_CHARS(i):
        r"""得到第 i 个 ``RAW_CHARS`` 的命名"""
        i = int(i) + 1
        return Const.RAW_CHAR + str(i)
    
    @staticmethod
    def INPUT_LENS(i):
        r"""得到第 i 个 ``INPUT_LEN`` 的命名"""
        i = int(i) + 1
        return Const.INPUT_LEN + str(i)
    
    @staticmethod
    def OUTPUTS(i):
        r"""得到第 i 个 ``OUTPUT`` 的命名"""
        i = int(i) + 1
        return Const.OUTPUT + str(i)
    
    @staticmethod
    def TARGETS(i):
        r"""得到第 i 个 ``TARGET`` 的命名"""
        i = int(i) + 1
        return Const.TARGET + str(i)
    
    @staticmethod
    def LOSSES(i):
        r"""得到第 i 个 ``LOSS`` 的命名"""
        i = int(i) + 1
        return Const.LOSS + str(i)
