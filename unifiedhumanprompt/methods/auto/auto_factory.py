import os.path
from typing import Any, Optional, OrderedDict, Type

from ...artifacts.artifact import get_config_file, get_prompt_file
from ...utils.config_utils import load_config
from ..base_method.method import PromptMethod


class BaseAutoMethod:
    # Base class for all auto methods
    _method_mapping: OrderedDict[str, Type[PromptMethod]] = None

    def __init__(self, *args: Any, **kwargs: Any):
        raise EnvironmentError("BaseAutoMethod is not meant to be instantiated")

    @classmethod
    # def from_config(cls, method_name=None, config_file_path=None, **kwargs):
    def from_config(
        cls,
        method_name: Optional[str] = None,
        config_file_path: Optional[str] = None,
        **kwargs: Any,
    ) -> PromptMethod:
        if method_name is None and config_file_path is None:
            raise ValueError(
                "You need to specify either a method name or a config file path."
            )

        # Fixme: why these pop and get things? do we need to fix the parameters num and category in config?
        if method_name is not None:

            # TODO: replace hard-coded path
            if "dataset_name" in kwargs:
                # Default to be examples/configs
                default_config_file_path = get_config_file(
                    f"{method_name}/{kwargs['dataset_name']}/config.yaml"
                )
            else:
                # If not found, default to be examples/configs
                default_config_file_path = get_config_file(
                    f"{method_name}/example/config.yaml"
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

        if "prompt_file_path" in config:
            config["prompt_file_path"] = get_prompt_file(config["prompt_file_path"])
        else:
            config["prompt_file_path"] = None

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
