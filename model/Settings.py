import yaml


class Settings:
    def __init__(self, configs_file: str):
        self.configs_file = configs_file
        self.configs = None

        self.load_configs()

    def load_configs(self):
        with open(self.configs_file, mode="r") as f:
            configs = yaml.safe_load(f)

        self.configs = configs
