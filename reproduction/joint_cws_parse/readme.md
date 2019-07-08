Code for paper [A Unified Model for Chinese Word Segmentation and Dependency Parsing](https://arxiv.org/abs/1904.04697)

### 准备数据
1. 数据应该为conll格式，1, 3, 6, 7列应该对应为'words', 'pos_tags', 'heads', 'labels'.
2. 将train, dev, test放在同一个folder下，并将该folder路径填入train.py中的data_folder变量里。
3. 从[百度云](https://pan.baidu.com/s/1uXnAZpYecYJITCiqgAjjjA)(提取:ua53)下载预训练vector，放到同一个folder下，并将train.py中vector_folder变量正确设置。


### 运行代码
```
python train.py 
```

### 其它
ctb5上跑出论文中报道的结果使用以上的默认参数应该就可以了(应该会更高一些); ctb7上使用默认参数会低0.1%左右，需要调节
learning rate scheduler. 