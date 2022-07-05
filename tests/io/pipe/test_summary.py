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

import pytest
import os

import pytest

from fastNLP.io import DataBundle
from fastNLP.io.pipe.summarization import ExtCNNDMPipe


class TestRunExtCNNDMPipe:

    def test_load(self):
        data_dir = 'tests/data_for_tests/io/cnndm'
        vocab_size = 100000
        VOCAL_FILE = 'tests/data_for_tests/io/cnndm/vocab'
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
        db = dbPipe.process_from_file(data_dir)
        db2 = dbPipe2.process_from_file(data_dir)

        assert(isinstance(db, DataBundle))
        assert(isinstance(db2, DataBundle))

        dbPipe3 = ExtCNNDMPipe(vocab_size=vocab_size,
                               sent_max_len=sent_max_len,
                               doc_max_timesteps=doc_max_timesteps,
                               domain=True)
        db3 = dbPipe3.process_from_file(data_dir)
        assert(isinstance(db3, DataBundle))

        with pytest.raises(RuntimeError):
            dbPipe4 = ExtCNNDMPipe(vocab_size=vocab_size,
                                   sent_max_len=sent_max_len,
                                   doc_max_timesteps=doc_max_timesteps)
            db4 = dbPipe4.process_from_file(os.path.join(data_dir, 'train.cnndm.jsonl'))

        dbPipe5 = ExtCNNDMPipe(vocab_size=vocab_size,
                               vocab_path=VOCAL_FILE,
                               sent_max_len=sent_max_len,
                               doc_max_timesteps=doc_max_timesteps,)
        db5 = dbPipe5.process_from_file(os.path.join(data_dir, 'train.cnndm.jsonl'))
        assert(isinstance(db5, DataBundle))

    def test_load_proc(self):
        data_dir = 'tests/data_for_tests/io/cnndm'
        vocab_size = 100000
        VOCAL_FILE = 'tests/data_for_tests/io/cnndm/vocab'
        sent_max_len = 100
        doc_max_timesteps = 50
        dbPipe = ExtCNNDMPipe(vocab_size=vocab_size,
                              vocab_path=VOCAL_FILE,
                              sent_max_len=sent_max_len,
                              doc_max_timesteps=doc_max_timesteps, num_proc=2)
        dbPipe2 = ExtCNNDMPipe(vocab_size=vocab_size,
                              vocab_path=VOCAL_FILE,
                              sent_max_len=sent_max_len,
                              doc_max_timesteps=doc_max_timesteps,
                                domain=True, num_proc=2)
        db = dbPipe.process_from_file(data_dir)
        db2 = dbPipe2.process_from_file(data_dir)

        assert(isinstance(db, DataBundle))
        assert(isinstance(db2, DataBundle))

        dbPipe3 = ExtCNNDMPipe(vocab_size=vocab_size,
                               sent_max_len=sent_max_len,
                               doc_max_timesteps=doc_max_timesteps,
                               domain=True, num_proc=2)
        db3 = dbPipe3.process_from_file(data_dir)
        assert(isinstance(db3, DataBundle))

        with pytest.raises(RuntimeError):
            dbPipe4 = ExtCNNDMPipe(vocab_size=vocab_size,
                                   sent_max_len=sent_max_len,
                                   doc_max_timesteps=doc_max_timesteps, num_proc=2)
            db4 = dbPipe4.process_from_file(os.path.join(data_dir, 'train.cnndm.jsonl'))

        dbPipe5 = ExtCNNDMPipe(vocab_size=vocab_size,
                               vocab_path=VOCAL_FILE,
                               sent_max_len=sent_max_len,
                               doc_max_timesteps=doc_max_timesteps, num_proc=2)
        db5 = dbPipe5.process_from_file(os.path.join(data_dir, 'train.cnndm.jsonl'))
        assert(isinstance(db5, DataBundle))


