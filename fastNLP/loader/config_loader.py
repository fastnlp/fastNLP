from loader.base_loader import BaseLoader


class ConfigLoader(BaseLoader):
    """loader for configuration files"""

    def __int__(self, data_name, data_path):
        super(ConfigLoader, self).__init__(data_name, data_path)
        self.config = self.parse(super(ConfigLoader, self).load())

    @staticmethod
    def parse(string):
        raise NotImplementedError
