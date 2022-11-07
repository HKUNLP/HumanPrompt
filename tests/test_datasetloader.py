from unifiedhumanprompt.tasks.dataset_loader import DatasetLoader


def test_init():
    dl = DatasetLoader.load_dataset("commonsense_qa")
    assert dl
