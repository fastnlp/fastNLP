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

import unittest
import os
# import sys
#
# sys.path.append("../../../")

from fastNLP.io import DataBundle
from fastNLP.io.pipe.summarization import ExtCNNDMPipe

class TestRunExtCNNDMPipe(unittest.TestCase):

    def test_load(self):
        data_set_dict = {
            'CNNDM': {"train": 'test/data_for_tests/cnndm.jsonl'},
        }
        vocab_size = 100000
        VOCAL_FILE = 'test/data_for_tests/cnndm.vocab'
        sent_max_len = 100
        doc_max_timesteps = 50
        dbPipe = ExtCNNDMPipe(vocab_size=vocab_size,
                              vocab_path=VOCAL_FILE,
                              sent_max_len=sent_max_len,
                              doc_max_timesteps=doc_max_timesteps)
        dbPipe2 = ExtCNNDMPipe(vocab_size=vocab_size,
                              vocab_path=VOCAL_FILE,
                              sent_max_len=sent_max_len,
                              doc_max_timesteps=doc_max_timesteps,
                                domain=True)
        for k, v in data_set_dict.items():
            db = dbPipe.process_from_file(v)
            db2 = dbPipe2.process_from_file(v)

            # print(db2.get_dataset("train"))

            self.assertTrue(isinstance(db, DataBundle))
            self.assertTrue(isinstance(db2, DataBundle))




