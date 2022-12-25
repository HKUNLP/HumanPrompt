import os
from typing import Any, Dict, Union

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
        "aqua": os.path.join(DIR, "aqua.py"),
        "add_sub": os.path.join(DIR, "add_sub.py"),
        "multi_arith": os.path.join(DIR, "multi_arith.py"),
        "wikitq": os.path.join(DIR, "wikitq.py"),
        "spider": os.path.join(DIR, "spider.py"),
    }

    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_split: str,
        dataset_subset_name: str = None,
        dataset_key_map: Dict[str, str] = None,
        **kwargs: Any
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        """
        Load dataset from the datasets library or from this repo.
        Args:
            dataset_name: name of the dataset
            dataset_split: split of the dataset
            dataset_subset_name: subset name of the dataset
            dataset_key_map: mapping original keys to a unified set of keys
            **kwargs: arguments to pass to the dataset

        Returns: dataset

        """
        # Check whether the dataset is in the own_dataset dictionary
        if dataset_name in DatasetLoader.own_dataset.keys():
            dataset_path = DatasetLoader.own_dataset[dataset_name]
            dataset = datasets.load_dataset(
                path=dataset_path,
                split=dataset_split,
                name=dataset_subset_name,
                **kwargs
            )
        else:
            # If not, load it from the datasets library
            dataset = datasets.load_dataset(
                path=dataset_name,
                split=dataset_split,
                name=dataset_subset_name,
                **kwargs
            )

        if dataset_key_map:
            reverse_dataset_key_map = {v: k for k, v in dataset_key_map.items()}
            dataset = dataset.rename_columns(reverse_dataset_key_map)

        return dataset
