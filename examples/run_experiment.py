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
    for data_item in dataset:
        # TODO: Add an adapter to ensure the `data_item` from different datasets has unified keys.
        data_item["context"] = "Answer choices: {}".format(
            " ".join(
                [
                    "({}) {}".format(label.lower(), text.lower())
                    for label, text in zip(
                        data_item["choices"]["label"], data_item["choices"]["text"]
                    )
                ]
            )
        )
        # TODO: Handle `None` prediction, cause error.
        prediction = method.run(data_item)
        gold_answer = data_item["answerKey"]
        # TODO: Maybe add an answer normalizer here, e.g., "(a)" equals to "A". Or relevant to the `transform`.
        prediction = prediction.lstrip("(").rstrip(")").lower()
        gold_answer = gold_answer.lower()
        predictions.append(prediction)
        gold_answers.append(gold_answer)
        print("*" * 80)
        print(f"pred answer: {prediction}")
        print(f"gold answer: {gold_answer}")

    eval_dict = evaluator.evaluate(predictions, gold_answers)
    return eval_dict


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"
    exp_config = load_config("configs/cot-commonsense_qa.yaml")

    dataset = DatasetLoader.load_dataset(
        dataset_name=exp_config["dataset"],
        split=exp_config["dataset_split"],
        name=exp_config["dataset_subset_name"]
        if "dataset_subset_name" in exp_config else None
    )

    if not hasattr(exp_config, "method"):
        raise ValueError("Experiment config must have a `method` field.")

    method_config = exp_config["method"]
    method = AutoMethod.from_config(
        method_name=method_config["method_name"]
        if method_config.get("method_name")
        else None,
        config_file_path=method_config["config_file_path"]
        if method_config.get("config_file_path")
        else None,
        **method_config.get("method_args", {}),
    )
    evaluator = Evaluator(exp_config["metrics"])

    eval_dict = run_experiment(dataset=dataset, method=method, evaluator=evaluator)
    print(eval_dict)
