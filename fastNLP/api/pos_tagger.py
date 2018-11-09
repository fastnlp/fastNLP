import pickle

import numpy as np

from fastNLP.core.dataset import DataSet
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.core.predictor import Predictor


class POS_tagger:
    def __init__(self):
        pass

    def predict(self, query):
        """ 
        :param query: List[str]
        :return answer: List[str]

        """
        # TODO: 根据query 构建DataSet
        pos_dataset = DataSet()
        pos_dataset["text_field"] = np.array(query)

        # 加载pipeline和model
        pipeline = self.load_pipeline("./xxxx")

        # 将DataSet作为参数运行 pipeline
        pos_dataset = pipeline(pos_dataset)

        # 加载模型
        model = ModelLoader().load_pytorch("./xxx")

        # 调 predictor
        predictor = Predictor()
        output = predictor.predict(model, pos_dataset)

        # TODO: 转成最终输出
        return None

    @staticmethod
    def load_pipeline(path):
        with open(path, "r") as fp:
            pipeline = pickle.load(fp)
        return pipeline
