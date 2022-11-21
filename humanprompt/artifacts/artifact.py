import os

import pkg_resources  # type: ignore


def get_config_file(config_path: str) -> str:
    """
    Returns path to a builtin config file.
    Args:
        config_path (str): config file name relative to humanprompt's "hub/"
            directory, e.g., "cot/commonsense_qa/config.yaml"
    Returns:
        str: the real path to the config file.
    """
    cfg_file = pkg_resources.resource_filename(
        "humanprompt.artifacts", os.path.join("hub", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))
    return cfg_file


def get_prompt_file(prompt_path: str) -> str:
    """
    Returns path to a builtin prompt file.
    Args:
        prompt_path (str): prompt file name relative to humanprompt's "prompts/"
            directory, e.g., "cot/commonsense_qa/prompt.txt"
    Returns:
        str: the real path to the prompt file.
    """
    prompt_file = pkg_resources.resource_filename(
        "humanprompt.artifacts", os.path.join("hub", prompt_path)
    )
    if not os.path.exists(prompt_file):
        raise RuntimeError("{} not available in Model Zoo!".format(prompt_path))
    return prompt_file
