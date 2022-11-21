import pytest

from humanprompt.methods.auto.method_auto import AutoMethod


@pytest.mark.usefixtures("openai_api_key")
def test_init() -> None:
    method = AutoMethod.from_config(method_name="cot")
    assert method


@pytest.mark.usefixtures("openai_api_key")
def test_run() -> None:
    method = AutoMethod.from_config("cot")
    prediction = method.run(
        {
            "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        }
    )
    assert prediction == "yes"
