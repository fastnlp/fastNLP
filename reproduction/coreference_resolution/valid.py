import torch
from reproduction.coreference_resolution.model.config import Config
from reproduction.coreference_resolution.model.metric import CRMetric
from fastNLP.io.pipe.coreference import CoReferencePipe

from fastNLP import Tester
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()
    
    config = Config()
    bundle = CoReferencePipe(Config()).process_from_file(
        {'train': config.train_path, 'dev': config.dev_path, 'test': config.test_path})
    metirc = CRMetric()
    model = torch.load(args.path)
    tester = Tester(bundle.get_dataset("test"),model,metirc,batch_size=1,device="cuda:0")
    tester.test()
    print('test over')


