#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys

sys.path.append('/remote-home/dqwang/FastNLP/fastNLP_brxx/')

from fastNLP.core.const import Const

from data.dataloader import SummarizationLoader
from tools.data import ExampleSet, Vocab

vocab_size = 100000
vocab_path = "test/testdata/vocab"
sent_max_len = 100
doc_max_timesteps = 50

# paths = {"train": "test/testdata/train.jsonl", "valid": "test/testdata/val.jsonl"}
paths = {"train": "/remote-home/dqwang/Datasets/CNNDM/train.label.jsonl", "valid": "/remote-home/dqwang/Datasets/CNNDM/val.label.jsonl"}
sum_loader = SummarizationLoader()
dataInfo = sum_loader.process(paths=paths, vocab_size=vocab_size, vocab_path=vocab_path, sent_max_len=sent_max_len, doc_max_timesteps=doc_max_timesteps, load_vocab_file=True)
trainset = dataInfo.datasets["train"]

vocab = Vocab(vocab_path, vocab_size)
dataset = ExampleSet(paths["train"], vocab, doc_max_timesteps, sent_max_len)

# print(trainset[0]["text"])
# print(dataset.get_example(0).original_article_sents)
# print(trainset[0]["words"])
# print(dataset[0][0].numpy().tolist())
b_size = len(trainset)
for i in range(b_size):
    if i <= 7327:
        continue
    print(trainset[i][Const.INPUT])
    print(dataset[i][0].numpy().tolist())
    assert trainset[i][Const.INPUT] == dataset[i][0].numpy().tolist(), i
    assert trainset[i][Const.INPUT_LEN] == dataset[i][2].numpy().tolist(), i
    assert trainset[i][Const.TARGET] == dataset[i][1].numpy().tolist(), i