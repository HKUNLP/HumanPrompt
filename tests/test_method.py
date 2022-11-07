import pytest

from unifiedhumanprompt.methods.auto.method_auto import AutoMethod


@pytest.mark.usefixtures("openai_api_key")
def test_init() -> None:
    method = AutoMethod.from_config(method_name="cot")
    assert method


@pytest.mark.usefixtures("openai_api_key")
def test_run() -> None:
    method = AutoMethod.from_config(method_name="cot")
    result = method.run(
        {
            "context": "Answer choices: (a) suburban development (b) apartment building (c) bus stop (d) michigan (e) suburbs",
            "question": "The townhouse was a hard sell for the realtor, it was right next to a high rise what?",
        }
    )
    assert result == "(b)"
