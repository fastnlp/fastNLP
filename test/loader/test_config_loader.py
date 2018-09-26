import configparser
import json
import os
import unittest

from fastNLP.loader.config_loader import ConfigSection, ConfigLoader


class TestConfigLoader(unittest.TestCase):
    def test_case_ConfigLoader(self):

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

        test_arg = ConfigSection()
        ConfigLoader("config").load_config(os.path.join("./test/loader", "config"), {"test": test_arg})

        section = read_section_from_config(os.path.join("./test/loader", "config"), "test")


        for sec in section:
            if (sec not in test_arg) or (section[sec] != test_arg[sec]):
                raise AttributeError("ERROR")

        for sec in test_arg.__dict__.keys():
            if (sec not in section) or (section[sec] != test_arg[sec]):
                raise AttributeError("ERROR")

        try:
            not_exist = test_arg["NOT EXIST"]
        except Exception as e:
            pass

        print("pass config test!")

