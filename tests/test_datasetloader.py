from humanprompt.tasks.dataset_loader import DatasetLoader


def test_init() -> None:
    dl = DatasetLoader.load_dataset("commonsense_qa")
    assert dl
