from typing import Union

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(file_path: str) -> Union[DictConfig, ListConfig]:
    config = OmegaConf.load(file_path)
    return config
