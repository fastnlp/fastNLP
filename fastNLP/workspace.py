import datetime
import os
import shutil

from colorama import Fore


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class WorkSpace(metaclass=Singleton):
    def __init__(self, force=False, backup=False):
        self.time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self._create_workspace(force, backup)

    def _create_workspace(self, force=False, backup=False):
        save_path = "./save/"
        if os.path.exists("./save"):
            print(Fore.RED + save_path + Fore.RESET + " already exists!")
            if not force:
                ans = input("Do you want to overwrite it? [y/N]:")
                if ans in ('y', 'Y', 'yes', 'Yes'):
                    if backup:
                        from getpass import getuser
                        tmp_path = "/tmp/{}-experiments/{}_{}".format(
                            getuser(), save_path, self.time_stamp)
                        print('move existing {} to {}'.format(
                            save_path, Fore.RED + tmp_path + Fore.RESET))
                        shutil.copytree(save_path, tmp_path)
                    shutil.rmtree(save_path)
            else:
                print("Overwrite it!")
                shutil.rmtree(save_path)
        if not os.path.exists("./save"):
            os.makedirs(save_path)
            print('create folder: ' + Fore.GREEN + save_path + Fore.RESET)

        if not os.path.exists("./model"):
            os.makedirs("./model")
            print('create folder: ' + Fore.GREEN + "./model" + Fore.RESET)
        os.makedirs(os.path.join("./model/", self.time_stamp))

        if not os.path.exists("./debug"):
            os.makedirs("./debug")
            print('create folder: ' + Fore.GREEN + "./debug" + Fore.RESET)
        os.makedirs(os.path.join("./debug/", self.time_stamp))

        if not os.path.exists("./log"):
            os.makedirs("./log")
            print('create folder: ' + Fore.GREEN + "./log" + Fore.RESET)

    @property
    def model_save_dir(self):
        return os.path.join("./model/", self.time_stamp)

    @property
    def debug_save_dir(self):
        return os.path.join("./debug/", self.time_stamp)

    @property
    def log_file_name(self):
        return os.path.join("./log/", self.time_stamp)
