import glob
import os
import shutil
from os import path
from typing import List

from setuptools import find_packages, setup

REQUIRES_PYTHON = ">=3.7.0"
NAME = "humanprompt"


def get_artifacts() -> List[str]:
    """
    Return a list of hub to include in package for model zoo. Copy over these hub inside
    humanprompt/artifacts.
    """

    # Use absolute paths while symlinking.
    source_hub_dir = path.join(path.dirname(path.realpath(__file__)), "hub")
    destination = path.join(
        path.dirname(path.realpath(__file__)),
        "humanprompt",
        "artifacts",
        "hub",
    )
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if path.exists(source_hub_dir):
        if path.islink(destination):
            os.unlink(destination)
        elif path.isdir(destination):
            shutil.rmtree(destination)

    if not path.exists(destination):
        try:
            os.symlink(source_hub_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_hub_dir, destination)

    config_paths = glob.glob("hub/**/**/*.yaml", recursive=True) + glob.glob(
        "hub/**/**/*.txt", recursive=True
    )
    return config_paths


install_requires = [
    "omegaconf",
    "datasets",
    "transformers",
    "evaluate",
    "scikit-learn",
    "manifest-ml",
    "backoff",
    "sqlparse",
]

extras_require = {
    "binder": [
        "binder@git+https://github.com/HKUNLP/Binder.git@humanprompt#egg=binder",
        "python-Levenshtein",
    ],
    "dev": [
        "black",
        "flake8",
        "isort",
        "mypy",
        "pytest",
        "pre-commit",
    ],
}

extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name=NAME,
    version="0.0.1",
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=("hub", "tests*")),
    package_data={"humanprompt.artifacts": get_artifacts()},
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
)
