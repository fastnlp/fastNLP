import configparser
import json
import os

from fastNLP.loader.base_loader import BaseLoader


class ConfigLoader(BaseLoader):
    """loader for configuration files"""

    def __int__(self, data_name, data_path):
        super(ConfigLoader, self).__init__(data_path)
        self.config = self.parse(super(ConfigLoader, self).load())

    @staticmethod
    def parse(string):
        raise NotImplementedError

    @staticmethod
    def load_config(file_path, sections):
        """
        :param file_path: the path of config file
        :param sections: the dict of {section_name(string): Section instance}
        Example:
            test_args = ConfigSection()
            ConfigLoader("config.cfg", "").load_config("./data_for_tests/config", {"POS_test": test_args})
        :return: return nothing, but the value of attributes are saved in sessions
        """
        assert isinstance(sections, dict)
        cfg = configparser.ConfigParser()
        if not os.path.exists(file_path):
            raise FileNotFoundError("config file {} not found. ".format(file_path))
        cfg.read(file_path)
        for s in sections:
            attr_list = [i for i in sections[s].__dict__.keys() if
                         not callable(getattr(sections[s], i)) and not i.startswith("__")]
            if s not in cfg:
                print('section %s not found in config file' % (s))
                continue
            gen_sec = cfg[s]
            for attr in gen_sec.keys():
                try:
                    val = json.loads(gen_sec[attr])
                    # print(s, attr, val, type(val))
                    if attr in attr_list:
                        assert type(val) == type(getattr(sections[s], attr)), \
                            'type not match, except %s but got %s' % \
                            (type(getattr(sections[s], attr)), type(val))
                    """
                            if attr in attr_list then check its type and
                        update its value.
                            else add a new attr in sections[s]
                    """
                    setattr(sections[s], attr, val)
                except Exception as e:
                    print("cannot load attribute %s in section %s"
                          % (attr, s))
                    pass


class ConfigSection(object):

    def __init__(self):
        pass

    def __getitem__(self, key):
        """
        :param key: str, the name of the attribute
        :return attr: the value of this attribute
            if key not in self.__dict__.keys():
                return self[key]
            else:
                raise AttributeError
        """
        if key in self.__dict__.keys():
            return getattr(self, key)
        raise AttributeError("do NOT have attribute %s" % key)

    def __setitem__(self, key, value):
        """
        :param key: str, the name of the attribute
        :param value: the value of this attribute
            if key not in self.__dict__.keys():
                self[key] will be added
            else:
                self[key] will be updated
        """
        if key in self.__dict__.keys():
            if not isinstance(value, type(getattr(self, key))):
                raise AttributeError("attr %s except %s but got %s" %
                                     (key, str(type(getattr(self, key))), str(type(value))))
        setattr(self, key, value)

    def __contains__(self, item):
        return item in self.__dict__.keys()

    @property
    def data(self):
        return self.__dict__


if __name__ == "__main__":
    config = ConfigLoader('there is no data')

    section = {'General': ConfigSection(), 'My': ConfigSection(), 'A': ConfigSection()}
    """
            General and My can be found in config file, so the attr and
        value will be updated
            A cannot be found in config file, so nothing will be done
    """

    config.load_config("../../test/data_for_tests/config", section)
    for s in section:
        print(s)
        for attr in section[s].__dict__.keys():
            print(s, attr, getattr(section[s], attr), type(getattr(section[s], attr)))
