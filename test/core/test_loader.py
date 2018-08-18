import os

import unittest
import json
import configparser

from fastNLP.loader.dataset_loader import POSDatasetLoader
from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.loader.preprocess import POSPreprocess


class TestPreprocess(unittest.TestCase):
    def test_case_1(self):
        data = [[["Tom", "and", "Jerry", "."], ["T", "F", "T", "F"]],
                ["Hello", "world", "!"], ["T", "F", "F"]]
        pickle_path = "./data_for_tests/"
        POSPreprocess(data, pickle_path)


class TestDatasetLoader(unittest.TestCase):
    def test_case_1(self):
        data = """Tom\tT\nand\tF\nJerry\tT\n.\tF\n\nHello\tT\nworld\tF\n!\tF"""
        lines = data.split("\n")
        answer = POSDatasetLoader.parse(lines)
        truth = [[["Tom", "and", "Jerry", "."], ["T", "F", "T", "F"]], [["Hello", "world", "!"], ["T", "F", "F"]]]
        self.assertListEqual(answer, truth, "POS Dataset Loader")


class TestConfigLoader(unittest.TestCase):
    def test_cast_1(self):
        def read_section_from_config(config_path, section_name):
            dict = {}
            if not os.path.exists(config_path):
                raise FileNotFoundError("config file {} NOT found.".format(config_path))
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            if section_name not in cfg:
                raise AttributeError("config file {} do NOT have section {}".format(
                    config_path, section_name
                ))
            gen_sec = cfg[section_name]
            for s in gen_sec.keys():
                try:
                    val = json.loads(gen_sec[s])
                    dict[s] = val
                except Exception as e:
                    raise AttributeError("json can NOT load {} in section {}, config file {}".format(
                        s, section_name, config_path
                    ))
            return dict

        train_args = ConfigSection()
        ConfigLoader("config.cfg", "").load_config("./data_for_tests/config", {"POS": train_args})
        dict = read_section_from_config("./data_for_tests/config", "POS")
        for sec in dict:
            if (sec not in train_args) or (dict[sec] != train_args[sec]):
                raise AttributeError("ERROR")
        for sec in train_args.__dict__.keys():
            if (sec not in dict) or (dict[sec] != train_args[sec]):
                raise AttributeError("ERROR")
        print("pass config test!")


if __name__ == '__main__':
    unittest.main()
