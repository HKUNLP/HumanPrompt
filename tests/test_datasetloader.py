from humanprompt.tasks.dataset_loader import DatasetLoader


def test_init() -> None:
    DatasetLoader.load_dataset("commonsense_qa", "train")
    DatasetLoader.load_dataset("commonsense_qa", "validation")
    DatasetLoader.load_dataset("commonsense_qa", "test")
    DatasetLoader.load_dataset("wikitq", "train")
    DatasetLoader.load_dataset("wikitq", "validation")
    DatasetLoader.load_dataset("wikitq", "test")
    DatasetLoader.load_dataset("spider", "train")
    DatasetLoader.load_dataset("spider", "validation")

    # todo: add more test
