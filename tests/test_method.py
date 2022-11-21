import pytest

from humanprompt.methods.auto.method_auto import AutoMethod


@pytest.mark.usefixtures("openai_api_key")
def test_init() -> None:
    method = AutoMethod.from_config(method_name="cot")
    assert method


@pytest.mark.usefixtures("openai_api_key")
def test_run() -> None:
    method = AutoMethod.from_config(
        method_name="cot",
        dataset_name="example",
        engine="code-davinci-002",
        temperature=0,
        stop_sequence="\n",
        transform="cot",
        extract="regex",
        extraction_regex="(?i).*So the answer is (.*).\n?",
        prompt_file_path="cot/example/prompt.txt",
        max_tokens=256,
    )
    prediction = method.run(
        {
            "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        }
    )
    assert prediction == "yes"
