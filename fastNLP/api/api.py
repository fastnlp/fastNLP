import torch
import warnings
warnings.filterwarnings('ignore')
import os

from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance

model_urls = {

}


class API:
    def __init__(self):
        self.pipeline = None

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, path):
        if os.path.exists(os.path.expanduser(path)):
            _dict = torch.load(path)
        else:
            _dict = load_url(path)
        self.pipeline = _dict['pipeline']


class POS_tagger(API):
    """FastNLP API for Part-Of-Speech tagging.

    """

    def __init__(self):
        super(POS_tagger, self).__init__()

    def predict(self, query):
        """

        :param query: list of list of str. Each string is a token(word).
        :return answer: list of list of str. Each string is a tag.
        """
        self.load("/home/zyfeng/fastnlp_0.2.0/reproduction/pos_tag_model/model_pp.pkl")

        data = DataSet()
        for example in query:
            data.append(Instance(words=example))

        out = self.pipeline(data)

        return [x["outputs"] for x in out]

    def load(self, name):
        _dict = torch.load(name)
        self.pipeline = _dict['pipeline']



class CWS(API):
    def __init__(self, model_path=None):
        super(CWS, self).__init__()
        if model_path is None:
            model_path = model_urls['cws']

        self.load(model_path)

    def predict(self, content):

        if not hasattr(self, 'pipeline'):
            raise ValueError("You have to load model first.")

        sentence_list = []
        # 1. 检查sentence的类型
        if isinstance(content, str):
            sentence_list.append(content)
        elif isinstance(content, list):
            sentence_list = content

        # 2. 组建dataset
        dataset = DataSet()
        dataset.add_field('raw_sentence', sentence_list)

        # 3. 使用pipeline
        self.pipeline(dataset)

        output = dataset['output'].content
        if isinstance(content, str):
            return output[0]
        elif isinstance(content, list):
            return output


if __name__ == "__main__":
    # tagger = POS_tagger()
    # print(tagger.predict([["我", "是", "学生", "。"], ["我", "是", "学生", "。"]]))

    cws = CWS()
    s = '编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。那么这款无人机到底有多厉害？是不是像它的外表那样神乎其神？未来无人机在战场上将发挥什么作用？本周《陈虎点兵》与您一起关注。　　本月12日，英国首次公布了最新研发的一款高科技无人驾驶隐身战机雷电之神。从外观上来看，这款无人机很有未来派的味道，全身融合，有点像飞碟，进气道也放在了飞机背部，一看就是具有很好的隐身性能。按照英国方面公布的情况，这款无人机是耗资相当于14.6亿元人民币，用了4年时间研发出来的。 　　雷电之神：大个头有大智慧　　目前关于这款无人机公布的信息还是比较含糊的，例如讲到了它的高速性能、洲际飞行能力，统统没有具体的数字。和现有或以前的一些无人机相比，这种无人机的特点主要有两个：　　第一，是高度的隐身。在此之前的无人战机也具备某种程度的隐身性能，但像雷电之神这样，全面运用隐身技术，从外形上看就具有高度隐形能力的无人机还是第一个。　　第二， 雷电之神的个头比较大。按照英国方面公布的数字，这架飞机的机长是11.35米，高3.98米，翼展将近10米，这个大小大概相当于英国的鹰式教练机和我们国产的L15高级教练机。按照英国人的说法这款无人机是世界最大，实际上肯定不是世界最大，因为它的尺寸比美国的全球鹰要小了不少，但在现有的无人机里，也算是大家伙了。大个头有大智慧，有大力量。它的尺寸决定了它具有较强的飞行能力和装载能力。按照英国人的说法，这款无人机具有洲际飞行能力，在飞行控制方面，可以通过卫星实现洲际飞行控制，这是在无人机控制，特别是远程控制上突破性的进展。这种飞机还配备了两个弹仓，可以进行攻击任务。 　　新一代无人机逐渐走向战场　　这些年来，无人机我们讲过不少，世界上推出的各种各样的无人机花样翻新，不断更新换代。为什么雷电之神值得我们去关注呢？我认为雷电之神本身的意义有限，但它标志着新一代的无人机开始逐渐走向战场，可能会掀起一个无人机的新时代。　　无人机从投入战场到现在，虽然时间很长，但真正引起大家关注、密集投入战斗使用的时间很短，从最早以色列在贝卡谷地使用无人机取得突出战绩，很快到了上世纪90年代末，美国推出了一系列新一代无人机，不过二十几年时间。无人机的发展速度非常快，进化能力很强，雷电之神的出现，使无人战机走进了一个新的时代。　　雷电之神的研制周期到目前为止只有4年，按照英国人公布的情况，2011年就要试飞。这个研制周期远远短于目前先进的有人战机的研制周期，这说明无人机的进化周期非常短，快速的进化使它在技术上能够迅速更新换代，作战能力和技术水平不断提高，以超越有人驾驶战机几倍的速度在发展。　　另外，这种无人机很便宜。我们知道研制三代机最少也要投入几百亿人民币，至于四代机、五代机，这个投入要更大。雷电之神到目前为止的投入仅为约14.6亿人民币，和有人驾驶高性能战机相比，便宜很多。　　从技术上来说，大家感觉无人机可能是个高科技的东西，实际上，无人机的技术门槛很低。我曾经接触过一些航空领域的专家，他们说无人机的进入门槛很低，所以很多企业和科研单位都在搞无人机，给人感觉是百花齐放，关键原因就是无人机较低的技术门槛。进化周期短，投入小，技术门槛低，这三个特点决定了无人机在未来一段时间将会快速的发展。 　　隐形无人机解决攻击航母的情报信息问题　　现在以雷电之神为代表的新一代无人机所表现出来的作战潜力，远远超过了之前的无人机。我们可以设想，像它这样高度隐身的无人机，在执行任务时可以神不知鬼不觉的进入你的防空圈。　　攻击航母很大程度上要取决于情报信息问题。像这种隐身无人机就可以实现神不知鬼不觉的跟踪航母，解决情报信息问题。　　从雷电之神的技术性能来看，它已经越来越接近于攻击型战斗机。看来无人机挑战传统空中力量这样的日子离我们越来越近了。这个问题应该是所有的国家和军队关注、关心的问题，如何应对这种挑战，如何在这种打破原有力量平衡的技术条件下，实现新的力量平衡，这是大家需要关注和研究的问题。新浪网'
    print(cws.predict([s]))

