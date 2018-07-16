import configparser
import json

from fastNLP.loader.base_loader import BaseLoader


class ConfigLoader(BaseLoader):
    """loader for configuration files"""

    def __int__(self, data_name, data_path):
        super(ConfigLoader, self).__init__(data_name, data_path)
        self.config = self.parse(super(ConfigLoader, self).load())

    @staticmethod
    def parse(string):
        raise NotImplementedError

    @staticmethod
    def load_config(file_path, sections):
        """
        :param file_path: the path of config file
        :param sections: the dict of sections
        :return:
        """
        cfg = configparser.ConfigParser()
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
                    #print(s, attr, val, type(val))
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
            if key not in self.__dict__.keys():
                return self[key]
            else:
                raise AttributeError
        """
        if key in self.__dict__.keys():
            return getattr(self, key)
        raise AttributeError('don\'t have attr %s' % (key))

    def __setitem__(self, key, value):
        """
            if key not in self.__dict__.keys():
                self[key] will be added
            else:
                self[key] will be updated
        """
        if key in self.__dict__.keys():
            if not type(value) == type(getattr(self, key)):
                raise AttributeError('attr %s except %s but got %s' % \
                                     (key, str(type(getattr(self, key))), str(type(value))))
        setattr(self, key, value)


if __name__ == "__name__":
    config = ConfigLoader('configLoader', 'there is no data')

    section = {'General': ConfigSection(), 'My': ConfigSection(), 'A': ConfigSection()}
    """
            General and My can be found in config file, so the attr and
        value will be updated
            A cannot be found in config file, so nothing will be done
    """

    config.load_config("config", section)
    for s in section:
        print(s)
        for attr in section[s].__dict__.keys():
            print(s, attr, getattr(section[s], attr), type(getattr(section[s], attr)))
    se = section['General']
    print(se["pre_trained"])
    se["pre_trained"] = False
    print(se["pre_trained"])
    #se["pre_trained"] = 5 #this will raise AttributeError: attr pre_trained except <class 'bool'> but got <class 'int'>