from omegaconf import OmegaConf


def load_config(file_path):
    config = OmegaConf.load(file_path)
    return config
