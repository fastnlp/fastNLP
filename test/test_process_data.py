from fastNLP.core.process_data import processData
#问题：一般类如ag_news_csv、yahoo_answers_csv等经常用于文本分类的数据集格式和fastNLP能够读取的格式并不兼容，通用的格式如
# yahoo_answers_csv中的一行数据（第一项是标签，第二项是title以及其他项的相关描述)：
# "4","What sea creature sleeps with one eye open?","","Dolphins."
# 而fastNLP能够处理的数据格式如下：
# But it does n't leave you with much .	1
# 解决方法：在Vocabulary增加一个字典保存函数和一个字典读取函数，而不是每次都生成一个新字典，同时减少下次运行的成本，第一次使用save_vocab()
# 生成字典后，下次可以直接使用load_vocab()载入的字典。

# 函数功能：将一些用于文本分类的主流数据集处理成fastNLP能够处理的格式并保存
if __name__ == '__main__':

   # 数据集d读取文件夹
   dataset_load_file = 'data_for_tests/ag_news_csv'
   # 数据集保存文件夹
   dataset_save_file = 'data_for_tests/ag_news_csv_fastNLP'
    # 数据集名称
   dataset_name = 'test.csv'
   processData(dataset_load_file, dataset_save_file, dataset_name)