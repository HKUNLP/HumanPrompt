import os.path

from ...artifacts.artifact import get_config_file, get_prompt_file
from ...utils.config_utils import load_config


class BaseAutoMethod:
    # Base class for all auto methods
    _method_mapping = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError("BaseAutoMethod is not meant to be instantiated")

    @classmethod
    def from_config(cls, method_name=None, config_file_path=None, **kwargs):
        if method_name is None and config_file_path is None:
            raise ValueError(
                "You need to specify either a method name or a config file path."
            )

        if method_name is not None:
            # TODO: replace hard-coded path
            default_config_file_path = get_config_file(
                f"{method_name}/configs/config.yaml"
            )
            if config_file_path is None:
                config_file_path = default_config_file_path
        if not os.path.exists(config_file_path):
            raise ValueError(f"Config file path {config_file_path} does not exist.")

        config = load_config(config_file_path)
        if "method_name" not in config:
            raise ValueError(
                "method_name must be specified in config file to instantiate a method."
            )

        config["prompt_file_path"] = get_prompt_file(config["prompt_file_path"])

        method_name = config["method_name"]
        method_cls = cls._method_mapping.get(method_name, None)
        if method_cls is None:
            raise ValueError(
                f"Unrecognized method name: {method_name}. "
                f"Available method names are: {list(cls._method_mapping.keys())}"
            )

        config.pop("method_name")
        method = method_cls(**config, **kwargs)
        return method
