# fastNLP 高级接口

### 环境与配置
1. 系统环境：linux/ubuntu(推荐)
2. 编程语言：Python>=3.6
3. Python包依赖
 - **torch==1.0**
 - numpy>=1.14.2

### 中文分词
```python
text = ['编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。',
        '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
        '那么这款无人机到底有多厉害？']
from fastNLP.api import CWS
cws = CWS(device='cpu')
print(cws.predict(text))
```

### 中文分词+词性标注
```python
text = ['编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。',
        '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
        '那么这款无人机到底有多厉害？']
from fastNLP.api import POS
pos = POS(device='cpu')
print(pos.predict(text))
```

### 中文分词+词性标注+句法分析
敬请期待

完整样例见`examples.py`