import os

from fastNLP.loader.config_loader import ConfigSection, ConfigLoader
from fastNLP.saver.logger import create_logger


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
        ConfigLoader(self.file_path).load_config(self.file_path, {sect_name: sect})
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
                log = create_logger(__name__, './config_saver.log')
                log.error("can NOT load config file [%s]" % self.file_path)
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
        if len(section_file.__dict__.keys()) == 0:#the section not in file before
            with open(self.file_path, 'a') as f:
                f.write('[' + section_name + ']\n')
                for k in section.__dict__.keys():
                    f.write(k + ' = ')
                    if isinstance(section[k], str):
                        f.write('\"' + str(section[k]) + '\"\n\n')
                    else:
                        f.write(str(section[k]) + '\n\n')
        else:
            change_file = False
            for k in section.__dict__.keys():
                if k not in section_file:
                    change_file = True
                    break
                if section_file[k] != section[k]:
                    logger = create_logger(__name__, "./config_loader.log")
                    logger.warning("section [%s] in config file [%s] has been changed" % (
                        section_name, self.file_path
                    ))
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
