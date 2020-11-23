import os
import unittest

from fastNLP.io import ModelSaver, ModelLoader
from fastNLP.models import CNNText


class TestModelIO(unittest.TestCase):
    def test_save_and_load(self):
        model = CNNText((10, 10), 2)
        saver = ModelSaver('tmp')
        loader = ModelLoader()
        saver.save_pytorch(model)
        
        new_cnn = CNNText((10, 10), 2)
        loader.load_pytorch(new_cnn, 'tmp')
        
        new_model = loader.load_pytorch_model('tmp')
        
        for i in range(10):
            for j in range(10):
                self.assertEqual(model.embed.embed.weight[i, j], new_cnn.embed.embed.weight[i, j])
                self.assertEqual(model.embed.embed.weight[i, j], new_model["embed.embed.weight"][i, j])
        
        os.system('rm tmp')
