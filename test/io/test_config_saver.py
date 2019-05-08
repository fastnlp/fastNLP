import os
import unittest

# from fastNLP.io import ConfigSection, ConfigLoader, ConfigSaver


class TestConfigSaver(unittest.TestCase):
    def test_case_1(self):
        config_file_dir = "."
        config_file_name = "config"
        config_file_path = os.path.join(config_file_dir, config_file_name)
        
        tmp_config_file_path = os.path.join(config_file_dir, "tmp_config")
        
        with open(config_file_path, "r") as f:
            lines = f.readlines()
        
        standard_section = ConfigSection()
        t_section = ConfigSection()
        ConfigLoader().load_config(config_file_path, {"test": standard_section, "t": t_section})
        
        config_saver = ConfigSaver(config_file_path)
        
        section = ConfigSection()
        section["doubles"] = 0.8
        section["tt"] = 0.5
        section["test"] = 105
        section["str"] = "this is a str"
        
        test_case_2_section = section
        test_case_2_section["double"] = 0.5
        
        for k in section.__dict__.keys():
            standard_section[k] = section[k]
        
        config_saver.save_config_file("test", section)
        config_saver.save_config_file("another-test", section)
        config_saver.save_config_file("one-another-test", section)
        config_saver.save_config_file("test-case-2", section)
        
        test_section = ConfigSection()
        at_section = ConfigSection()
        another_test_section = ConfigSection()
        one_another_test_section = ConfigSection()
        a_test_case_2_section = ConfigSection()
        
        ConfigLoader().load_config(config_file_path, {"test": test_section,
                                                      "another-test": another_test_section,
                                                      "t": at_section,
                                                      "one-another-test": one_another_test_section,
                                                      "test-case-2": a_test_case_2_section})
        
        assert test_section == standard_section
        assert at_section == t_section
        assert another_test_section == section
        assert one_another_test_section == section
        assert a_test_case_2_section == test_case_2_section
        
        config_saver.save_config_file("test", section)
        
        with open(config_file_path, "w") as f:
            f.writelines(lines)
        
        with open(tmp_config_file_path, "w") as f:
            f.write('[test]\n')
            f.write('this is an fault example\n')
        
        tmp_config_saver = ConfigSaver(tmp_config_file_path)
        try:
            tmp_config_saver._read_section()
        except Exception as e:
            pass
        os.remove(tmp_config_file_path)
        
        try:
            tmp_config_saver = ConfigSaver("file-NOT-exist")
        except Exception as e:
            pass
    
    def test_case_2(self):
        config = "[section_A]\n[section_B]\n"
        
        with open("./test.cfg", "w", encoding="utf-8") as f:
            f.write(config)
        saver = ConfigSaver("./test.cfg")
        
        section = ConfigSection()
        section["doubles"] = 0.8
        section["tt"] = [1, 2, 3]
        section["test"] = 105
        section["str"] = "this is a str"
        
        saver.save_config_file("section_A", section)
        
        os.system("rm ./test.cfg")
    
    def test_case_3(self):
        config = "[section_A]\ndoubles = 0.9\ntt = [1, 2, 3]\n[section_B]\n"
        
        with open("./test.cfg", "w", encoding="utf-8") as f:
            f.write(config)
        saver = ConfigSaver("./test.cfg")
        
        section = ConfigSection()
        section["doubles"] = 0.8
        section["tt"] = [1, 2, 3]
        section["test"] = 105
        section["str"] = "this is a str"
        
        saver.save_config_file("section_A", section)
        
        os.system("rm ./test.cfg")
