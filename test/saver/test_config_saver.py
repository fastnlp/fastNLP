import os

import unittest

from fastNLP.loader.config_loader import ConfigSection, ConfigLoader
from fastNLP.saver.config_saver import ConfigSaver


class TestConfigSaver(unittest.TestCase):
    def test_case_1(self):
        config_saver = ConfigSaver("./test/loader/config")
        #config_saver = ConfigSaver("./../loader/config")

        section = ConfigSection()
        section["test"] = 105
        section["tt"] = 0.5
        section["str"] = "this is a str"
        config_saver.save_config_file("test", section)
        config_saver.save_config_file("another-test", section)
        config_saver.save_config_file("one-another-test", section)

