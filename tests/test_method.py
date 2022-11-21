import pytest

from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.tasks.dataset_loader import DatasetLoader


@pytest.mark.usefixtures("openai_api_key")
def test_init() -> None:
    method = AutoMethod.from_config("cot")
    # todo, add all methods
    assert method


@pytest.mark.usefixtures("openai_api_key")
def test_run() -> None:
    # todo, add all methods

    # CoT
    # commonsense_qa
    method = AutoMethod.from_config("cot")
    data_item = DatasetLoader.load_dataset(
        dataset_name="commonsense_qa",
        dataset_split="validation",
        dataset_key_map={
            "question": "question",
            "choices": "choices",
            "answer": "answerKey",
            "id": "id",
        },
    )[0]
    method.run(data_item)
