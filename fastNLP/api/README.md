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
# ['编者 按 : 7月 12日 , 英国 航空 航天 系统 公司 公布 了 该 公司 研制 的 第一 款 高 科技 隐形 无人 机雷电 之 神 。', '这 款 飞行 从 外型 上 来 看 酷似 电影 中 的 太空 飞行器 , 据 英国 方面 介绍 , 可以 实现 洲际 远程 打击 。', '那么 这 款 无人 机 到底 有 多 厉害 ?']
```

### 中文分词+词性标注
```python
text = ['编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。',
        '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
        '那么这款无人机到底有多厉害？']
from fastNLP.api import POS
pos = POS(device='cpu')
print(pos.predict(text))
# [['编者/NN', '按/P', '：/PU', '7月/NT', '12日/NR', '，/PU', '英国/NR', '航空/NN', '航天/NN', '系统/NN', '公司/NN', '公布/VV', '了/AS', '该/DT', '公司/NN', '研制/VV', '的/DEC', '第一/OD', '款高/NN', '科技/NN', '隐形/NN', '无/VE', '人机/NN', '雷电/NN', '之/DEG', '神/NN', '。/PU'], ['这/DT', '款/NN', '飞行/VV', '从/P', '外型/NN', '上/LC', '来/MSP', '看/VV', '酷似/VV', '电影/NN', '中/LC', '的/DEG', '太空/NN', '飞行器/NN', '，/PU', '据/P', '英国/NR', '方面/NN', '介绍/VV', '，/PU', '可以/VV', '实现/VV', '洲际/NN', '远程/NN', '打击/NN', '。/PU'], ['那么/AD', '这/DT', '款/NN', '无/VE', '人机/NN', '到底/AD', '有/VE', '多/CD', '厉害/NN', '？/PU']]
```

### 中文分词+词性标注+句法分析
```python
text = ['编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。',
        '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
        '那么这款无人机到底有多厉害？']
from fastNLP.api import Parser
parser = Parser(device='cpu')
print(parser.predict(text))
# [['12/nsubj', '12/prep', '2/punct', '5/nn', '2/pobj', '12/punct', '11/nn', '11/nn', '11/nn', '11/nn', '2/pobj', '0/root', '12/asp', '15/det', '16/nsubj', '21/rcmod', '16/cpm', '21/nummod', '21/nn', '21/nn', '22/top', '12/ccomp', '24/nn', '26/assmod', '24/assm', '22/dobj', '12/punct'], ['2/det', '8/xsubj', '8/mmod', '8/prep', '6/lobj', '4/plmod', '8/prtmod', '0/root', '8/ccomp', '11/lobj', '14/assmod', '11/assm', '14/nn', '9/dobj', '8/punct', '22/prep', '18/nn', '19/nsubj', '16/pccomp', '22/punct', '22/mmod', '8/dep', '25/nn', '25/nn', '22/dobj', '8/punct'], ['4/advmod', '3/det', '4/nsubj', '0/root', '4/dobj', '7/advmod', '4/conj', '9/nummod', '7/dobj', '4/punct']]
```

完整样例见`examples.py`