import os

import pkg_resources


def get_config_file(config_path):
    """
    Returns path to a builtin config file.
    Args:
        config_path (str): config file name relative to unifiedhumanprompt's "configs/"
            directory, e.g., "cot/configs/config.yaml"
    Returns:
        str: the real path to the config file.
    """
    cfg_file = pkg_resources.resource_filename(
        "unifiedhumanprompt.artifacts", os.path.join("configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))
    return cfg_file


def get_prompt_file(prompt_path):
    """
    Returns path to a builtin prompt file.
    Args:
        prompt_path (str): prompt file name relative to unifiedhumanprompt's "prompts/"
            directory, e.g., "cot/prompt_files/commonsense_qa.txt"
    Returns:
        str: the real path to the prompt file.
    """
    prompt_file = pkg_resources.resource_filename(
        "unifiedhumanprompt.artifacts", os.path.join("configs", prompt_path)
    )
    if not os.path.exists(prompt_file):
        raise RuntimeError("{} not available in Model Zoo!".format(prompt_path))
    return prompt_file
