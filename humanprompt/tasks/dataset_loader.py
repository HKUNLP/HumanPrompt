import os
from typing import Any, Union, List, Dict

import datasets
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

DIR = os.path.join(os.path.dirname(__file__))


class DatasetLoader(object):
    """Dataset loader class."""

    # Datasets implemented in this repo
    # Either they do not exist in the datasets library or they have something improper
    own_dataset = {
        "strategy_qa": os.path.join(DIR, "strategy_qa.py"),
        "svamp": os.path.join(DIR, "svamp.py"),
    }

    @staticmethod
    def load_dataset(
            dataset_name: str,
            **kwargs: Any
    ) -> Union[datasets.Dataset, List, Dict]:
        """
        Load dataset from the datasets library or from this repo.
        Args:
            dataset_name: name of the dataset dd
            **kwargs: arguments to pass to the dataset

        Returns: dataset

        """
        # Check whether the dataset is in the own_dataset dictionary
        if dataset_name in DatasetLoader.own_dataset.keys():
            dataset_path = DatasetLoader.own_dataset[dataset_name]
            return datasets.load_dataset(dataset_path, **kwargs)
        else:
            # If not, load it from the datasets library
            return datasets.load_dataset(dataset_name, **kwargs)
