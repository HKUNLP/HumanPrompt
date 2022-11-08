import os
from typing import Dict, List, Union

from unifiedhumanprompt.evaluators.evaluator import Evaluator
from unifiedhumanprompt.methods.auto.method_auto import AutoMethod
from unifiedhumanprompt.methods.base_method.method import PromptMethod
from unifiedhumanprompt.tasks.dataset_loader import DatasetLoader
from unifiedhumanprompt.utils.config_utils import load_config


def run_experiment(
    dataset: Union[Dict, List],
    method: PromptMethod,
    evaluator: Evaluator,
) -> Dict:
    """
    Run experiment on a dataset using a method.

    Args:
        dataset: Dataset to run experiment on.
        method: Method to run experiment with.
        evaluator: Evaluator to evaluate the experiment performance.
    """
    # TODO: Add wandb logging
    predictions, gold_answers = [], []
    for id, data_item in enumerate(dataset):
        model_input = {
            "question": data_item["question"],
        }
        prediction = method.run(model_input)
        gold_answer = data_item["answer"]
        predictions.append({"prediction_text": prediction, "id": f"test-{id}"})
        gold_answers.append(
            {
                "answers": {"answer_start": [0], "text": [gold_answer]},
                "id": f"test-{id}",
            }
        )
        print("*" * 80)
        print(f"pred answer: {prediction}")
        print(f"gold answer: {gold_answer}")

    eval_dict = evaluator.evaluate(predictions, gold_answers)
    return eval_dict


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"
    exp_config = load_config("configs/cot-hotpotqa.yaml")
    if not hasattr(exp_config, "method"):
        raise ValueError("Experiment config must have a `method` field.")

    method_config = exp_config["method"]
    dataset = DatasetLoader.load_dataset(
        dataset_name=exp_config["dataset"], name="fullwiki"
    )
    if exp_config["dataset_split"] not in dataset:
        raise ValueError(
            f"Dataset {exp_config['dataset']} does not have split {exp_config['dataset_split']}."
        )

    dataset = dataset[exp_config["dataset_split"]]
    max_test_samples = exp_config.get("max_test_samples", None)
    if max_test_samples:
        dataset = dataset.select(range(max_test_samples))

    method = AutoMethod.from_config(
        method_name=method_config["method_name"]
        if method_config.get("method_name")
        else None,
        config_file_path=method_config["config_file_path"]
        if method_config.get("config_file_path")
        else None,
        dataset_name=exp_config["dataset"],
        **method_config.get("method_args", {}),
    )
    evaluator = Evaluator(exp_config["metrics"])

    eval_dict = run_experiment(dataset=dataset, method=method, evaluator=evaluator)
    print(eval_dict)
