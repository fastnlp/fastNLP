import os
import unittest

from fastNLP.workspace import WorkSpace


class TestWorkSpace(unittest.TestCase):
    def test(self):
        os.makedirs("test", exist_ok=True)
        os.system("cd test")
        ws = WorkSpace(force=True)

        self.assertTrue(os.path.exists("model"))
        self.assertTrue(os.path.exists("debug"))
        self.assertTrue(os.path.exists("log"))

        os.system("cd ..")
        os.rmdir("test")
