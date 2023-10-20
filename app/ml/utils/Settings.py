"""Model Settings"""

import yaml

class Settings:
    """Settings of model"""
    def __init__(self, configs_file: str):
        self.configs_file = configs_file
        self.configs = None

        self.load_configs()

    def load_configs(self):
        """Load config from .yaml"""
        with open(self.configs_file, mode="r", encoding="UTF-8") as f:
            configs = yaml.safe_load(f)

        self.configs = configs
