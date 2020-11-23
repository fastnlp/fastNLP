from fastNLP import logger
import unittest
from unittest.mock import  patch
import os
import io
import tempfile
import shutil

class TestLogger(unittest.TestCase):
    msg = 'some test logger msg'

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        pass
        # shutil.rmtree(self.tmpdir)

    def test_add_file(self):
        fn = os.path.join(self.tmpdir, 'log.txt')
        logger.add_file(fn)
        logger.info(self.msg)
        with open(fn, 'r') as f:
            line = ''.join([l for l in f])
            print(line)
        self.assertTrue(self.msg in line)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_stdout(self, mock_out):
        for i in range(3):
            logger.info(self.msg)
            logger.debug('aabbc')

        self.assertEqual([self.msg for i in range(3)], mock_out.getvalue().strip().split('\n'))
