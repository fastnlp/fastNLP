import configparser
import json
import os

from fastNLP.io.base_loader import BaseLoader


class ConfigLoader(BaseLoader):
    """loader for configuration files"""

    def __init__(self, data_path=None):
        super(ConfigLoader, self).__init__()
        if data_path is not None:
            self.config = self.parse(super(ConfigLoader, self).load(data_path))

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
        """
        :param item: The key of item.
        :return: True if the key in self.__dict__.keys() else False.
        """
        return item in self.__dict__.keys()

    def __eq__(self, other):
        """Overwrite the == operator

        :param other: Another ConfigSection() object which to be compared.
        :return: True if value of each key in each ConfigSection() object are equal to the other, else False.
        """
        for k in self.__dict__.keys():
            if k not in other.__dict__.keys():
                return False
            if getattr(self, k) != getattr(self, k):
                return False

        for k in other.__dict__.keys():
            if k not in self.__dict__.keys():
                return False
            if getattr(self, k) != getattr(self, k):
                return False

        return True

    def __ne__(self, other):
        """Overwrite the != operator

        :param other:
        :return:
        """
        return not self.__eq__(other)

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


class ConfigSaver(object):

    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("file {} NOT found!".__format__(self.file_path))

    def _get_section(self, sect_name):
        """This is the function to get the section with the section name.

        :param sect_name: The name of section what wants to load.
        :return: The section.
        """
        sect = ConfigSection()
        ConfigLoader().load_config(self.file_path, {sect_name: sect})
        return sect

    def _read_section(self):
        """This is the function to read sections from the config file.

        :return: sect_list, sect_key_list
            sect_list: A list of ConfigSection().
            sect_key_list: A list of names in sect_list.
        """
        sect_name = None

        sect_list = {}
        sect_key_list = []

        single_section = {}
        single_section_key = []

        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('[') and line.endswith(']\n'):
                if sect_name is None:
                    pass
                else:
                    sect_list[sect_name] = single_section, single_section_key
                    single_section = {}
                    single_section_key = []
                    sect_key_list.append(sect_name)
                sect_name = line[1: -2]
                continue

            if line.startswith('#'):
                single_section[line] = '#'
                single_section_key.append(line)
                continue

            if line.startswith('\n'):
                single_section_key.append('\n')
                continue

            if '=' not in line:
                # log = create_logger(__name__, './config_saver.log')
                # log.error("can NOT load config file [%s]" % self.file_path)
                raise RuntimeError("can NOT load config file {}".__format__(self.file_path))

            key = line.split('=', maxsplit=1)[0].strip()
            value = line.split('=', maxsplit=1)[1].strip() + '\n'
            single_section[key] = value
            single_section_key.append(key)

        if sect_name is not None:
            sect_list[sect_name] = single_section, single_section_key
            sect_key_list.append(sect_name)
        return sect_list, sect_key_list

    def _write_section(self, sect_list, sect_key_list):
        """This is the function to write config file with section list and name list.

        :param sect_list: A list of ConfigSection() need to be writen into file.
        :param sect_key_list: A list of name of sect_list.
        :return:
        """
        with open(self.file_path, 'w') as f:
            for sect_key in sect_key_list:
                single_section, single_section_key = sect_list[sect_key]
                f.write('[' + sect_key + ']\n')
                for key in single_section_key:
                    if key == '\n':
                        f.write('\n')
                        continue
                    if single_section[key] == '#':
                        f.write(key)
                        continue
                    f.write(key + ' = ' + single_section[key])
                f.write('\n')

    def save_config_file(self, section_name, section):
        """This is the function to be called to change the config file with a single section and its name.

        :param section_name: The name of section what needs to be changed and saved.
        :param section: The section with key and value what needs to be changed and saved.
        :return:
        """
        section_file = self._get_section(section_name)
        if len(section_file.__dict__.keys()) == 0:  # the section not in the file before
            # append this section to config file
            with open(self.file_path, 'a') as f:
                f.write('[' + section_name + ']\n')
                for k in section.__dict__.keys():
                    f.write(k + ' = ')
                    if isinstance(section[k], str):
                        f.write('\"' + str(section[k]) + '\"\n\n')
                    else:
                        f.write(str(section[k]) + '\n\n')
        else:
            # the section exists
            change_file = False
            for k in section.__dict__.keys():
                if k not in section_file:
                    # find a new key in this section
                    change_file = True
                    break
                if section_file[k] != section[k]:
                    # logger = create_logger(__name__, "./config_loader.log")
                    # logger.warning("section [%s] in config file [%s] has been changed" % (
                    #    section_name, self.file_path
                    # ))
                    change_file = True
                    break
            if not change_file:
                return

            sect_list, sect_key_list = self._read_section()
            if section_name not in sect_key_list:
                raise AttributeError()

            sect, sect_key = sect_list[section_name]
            for k in section.__dict__.keys():
                if k not in sect_key:
                    if sect_key[-1] != '\n':
                        sect_key.append('\n')
                    sect_key.append(k)
                sect[k] = str(section[k])
                if isinstance(section[k], str):
                    sect[k] = "\"" + sect[k] + "\""
                sect[k] = sect[k] + "\n"
            sect_list[section_name] = sect, sect_key
            self._write_section(sect_list, sect_key_list)
