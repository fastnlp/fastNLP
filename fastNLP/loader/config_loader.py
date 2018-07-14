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

if __name__ == "__name__":
    config = ConfigLoader('configLoader', 'there is no data')


    class ConfigSection(object):
        def __init__(self):
            pass


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
            print(s, attr, getattr(section[s], attr))