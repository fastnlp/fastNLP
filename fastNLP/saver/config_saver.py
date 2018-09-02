import os

import json
import configparser

from fastNLP.loader.config_loader import ConfigSection, ConfigLoader
from fastNLP.saver.logger import create_logger

class ConfigSaver(object):

    def __init__(self, file_path):
        self.file_path = file_path

    def save_section(self, section_name, section):
        cfg = configparser.ConfigParser()
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("config file {} not found. ".format(self.file_path))
        cfg.read(self.file_path)
        if section_name not in cfg:
            cfg.add_section(section_name)
        gen_sec = cfg[section_name]
        for key in section:
            if key in gen_sec.keys():
                try:
                    val = json.load(gen_sec[key])
                except Exception as e:
                    print("cannot load attribute %s in section %s"
                          % (key, section_name))
                try:
                    assert section[key] == val
                except Exception as e:
                    logger = create_logger(__name__, "./config_saver.log")
                    logger.warning("this is a warning #TODO")
            cfg.set(section_name,key, section[key])
        cfg.write(open(self.file_path, 'w'))
