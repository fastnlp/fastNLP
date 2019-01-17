import os
import pandas as pd
from collections import defaultdict
import numpy as np
#功能：将特定的数据集格式处理成fastNLP能够读取的格式并保存
# 问题：fastNLP虽然已经提供了split函数，可以将数据集划分成训练集和测试机，但一般网上用作训练的标准集都已经提前划分好了训练集和测试机，
# 而使用split将数据集进行随机划分还引来了一个问题：
#       因为每次都是随机划分，导致每次的字典都不一样，保存好模型下次再载入进行测试时，结果差异非常大。
#
# 解决方法：在Vocabulary增加一个字典保存函数和一个字典读取函数，而不是每次都生成一个新字典，同时减少下次运行的成本，第一次使用save_vocab()
# 生成字典后，下次可以直接使用load_vocab()载入的字典。

#测试：在test文件夹下有test_process_data用于测试
def processData(dataset_load_file,dataset_save_file,dataset_name):
    # load train data
    print("start load data.......")

    train_data_df = pd.read_csv(os.path.join(dataset_load_file, dataset_name), header=None)
    print('shape(train_data_df.iloc)',train_data_df.shape)
    rows=train_data_df.shape[0]
    columns = train_data_df.shape[1]
    #
    label=train_data_df.iloc[:, 0]
    title = train_data_df.iloc[:, 1]

    list = []
    for i in range(0, rows):
      context = title[i]
      for j in range(2, columns):
        # 排除引号内容为空的情况
        if not pd.isnull(train_data_df.iloc[:, j][i]):
          context = context + "  " + train_data_df.iloc[:, j][i]
      # context.replace('\n','').replace('"','').replace("\r",'')
      context=str(context)+" \t"+str(label[i])+"\n"
      list.append(context)


    dataframe = pd.DataFrame(data=list)
    dataframe.to_csv(os.path.join(dataset_save_file, dataset_name), index=False, header=None)

    print('finish!')
    return train_data_df,label,list

#功能：将特定的数据集格式转换成fastNLP可以处理的格式并保存

if __name__ == '__main__':

   # 数据集保存文件夹
   dataset_load_file = 'data/ag_news_csv'
   dataset_save_file = 'data/ag_news_csv_fastNLP'
    # 数据集名称
   dataset_name = 'test.csv'
   processData(dataset_load_file, dataset_save_file, dataset_name)






