import glob
import os
import shutil
from os import path
from typing import List

from setuptools import find_packages, setup

REQUIRES_PYTHON = ">=3.7.0"
NAME = "unifiedhumanprompt"


def get_artifacts() -> List[str]:
    """
    Return a list of configs to include in package for model zoo. Copy over these configs inside
    unifiedhumanprompt/artifacts.
    """

    # Use absolute paths while symlinking.
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
    destination = path.join(
        path.dirname(path.realpath(__file__)),
        "unifiedhumanprompt",
        "artifacts",
        "configs",
    )
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if path.exists(source_configs_dir):
        if path.islink(destination):
            os.unlink(destination)
        elif path.isdir(destination):
            shutil.rmtree(destination)

    if not path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob("configs/**/*.yaml", recursive=True) + glob.glob(
        "configs/**/*.txt", recursive=True
    )
    return config_paths


setup(
    name=NAME,
    version="0.0.1",
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data={"unifiedhumanprompt.artifacts": get_artifacts()},
    include_package_data=True,
)
