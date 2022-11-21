import os.path
from typing import Any, Optional, OrderedDict, Type

from omegaconf.dictconfig import DictConfig

from ...artifacts.artifact import get_config_file, get_prompt_file
from ...utils.config_utils import load_config
from ..base_method.method import PromptMethod


class BaseAutoMethod:
    # Base class for all auto methods
    _method_mapping: OrderedDict[str, Type[PromptMethod]] = None

    def __init__(self, *args: Any, **kwargs: Any):
        raise EnvironmentError("BaseAutoMethod is not meant to be instantiated")

    @classmethod
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
            if "dataset_name" in kwargs:
                # Default to be examples/hub
                default_config_file_path = get_config_file(
                    f"{method_name}/{kwargs['dataset_name']}/config.yaml"
                )
            else:
                # If not found, default to be examples/hub
                default_config_file_path = get_config_file(
                    f"{method_name}/example/config.yaml"
                )
            if config_file_path is None:
                config_file_path = default_config_file_path
        if not os.path.exists(config_file_path):
            raise ValueError(f"Config file path {config_file_path} does not exist.")

        config = load_config(config_file_path)
        assert isinstance(config, DictConfig), "Only DictConfig is supported for now."

        config.update(kwargs)
        if "method_name" not in config:
            raise ValueError(
                "method_name must be specified in config file to instantiate a method."
            )

        if "prompt_file_path" in config:
            config["prompt_file_path"] = get_prompt_file(config["prompt_file_path"])

        if "prompt_examples_path" in config:
            if isinstance(config["prompt_examples_path"], str):
                config["prompt_examples_path"] = get_prompt_file(
                    config["prompt_examples_path"]
                )
            elif isinstance(config["prompt_examples_path"], DictConfig):
                for key in config["prompt_examples_path"]:
                    config["prompt_examples_path"][key] = get_prompt_file(
                        config["prompt_examples_path"][key]
                    )
            else:
                raise ValueError("prompt_examples_path must be a string or a dict")

        method_name = config["method_name"]
        method_cls = cls._method_mapping.get(method_name, None)
        if method_cls is None:
            raise ValueError(
                f"Unrecognized method name: {method_name}. "
                f"Available method names are: {list(cls._method_mapping.keys())}"
            )

        method = method_cls(**config)
        return method
