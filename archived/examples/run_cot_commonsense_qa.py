import json
import os
import time
from typing import Dict, List

import openai
from datasets import Dataset

from humanprompt.evaluators.evaluator import Evaluator
from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.methods.base_method.method import PromptMethod
from humanprompt.tasks.dataset_loader import DatasetLoader
from humanprompt.utils.config_utils import load_config


class OpenAIKeyPool:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.idx = 0

    def get_key(self) -> str:
        key = self.keys[self.idx]
        self.idx += 1
        if self.idx == len(self.keys):
            self.idx = 0
        return key


def run_experiment(
    dataset: Dataset,
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
    predictions, gold_answers = [], []
    for idx, data_item in enumerate(dataset):
        if data_item.get('id', None) is None:
            data_item['id'] = idx

        if os.path.exists(os.path.join(tmp_save_dir, f"{idx}_{data_item['id']}.json")):
            # Already inferenced example
            with open(
                os.path.join(tmp_save_dir, f"{idx}_{data_item['id']}.json"), "r"
            ) as f:
                result_item = json.load(f)
                prediction, gold_answer = (
                    result_item["prediction"],
                    result_item["gold_answer"],
                )
        else:
            # New coming example
            while True:
                try:
                    current_key = openai_key_pool.get_key()
                    os.environ["OPENAI_API_KEY"] = current_key
                    print("Using OpenAI key: ", current_key)
                    prediction = method.run(x=data_item, verbose=verbose)
                    break
                except openai.error.OpenAIError as e:
                    print(f"Error when getting response: {e}")
                    continue
            if prediction is None:
                prediction = "<None>"
            gold_answer = data_item["answer"].lower()
            # Cache current example
            os.makedirs(tmp_save_dir, exist_ok=True)
            with open(
                os.path.join(tmp_save_dir, f"{idx}_{data_item['id']}.json"), "w"
            ) as f:
                json.dump(
                    {
                        "idx": idx,
                        "id": data_item["id"],
                        "prediction": prediction,
                        "gold_answer": gold_answer,
                    },
                    f,
                )
        predictions.append(prediction)
        gold_answers.append(gold_answer)
        print(f"idx: {idx}")
        print(f"id: {data_item['id']}")
        print(f"pred answer: {prediction}")
        print(f"gold answer: {gold_answer}")
        print()
    # Evaluate
    eval_dict = evaluator.evaluate(predictions, gold_answers)
    return eval_dict


if __name__ == "__main__":
    # Meta-config
    start_time = time.time()
    openai_key_pool = OpenAIKeyPool(
        keys=[
            "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",
            "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",
            "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",
            "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",
            "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",
        ]
    )
    os.environ["OPENAI_API_KEY"] = openai_key_pool.get_key()
    verbose = False
    exp_name = "cot-svamp"
    exp_config = load_config(f"configs/{exp_name}.yaml")
    save_dir = "results/"
    tmp_save_dir = os.path.join(save_dir, "tmp", f"{exp_name}/")

    # Config
    if not hasattr(exp_config, "dataset"):
        raise ValueError("Experiment config must have a `dataset` field.")

    dataset_config = exp_config["dataset"]
    dataset = DatasetLoader.load_dataset(
        dataset_name=dataset_config["dataset_name"],
        dataset_split=dataset_config["dataset_split"],
        dataset_subset_name=dataset_config["dataset_subset_name"]
        if "dataset_subset_name" in dataset_config
        else None,
        dataset_key_map=dataset_config["dataset_key_map"]
        if "dataset_key_map" in dataset_config
        else None,
    )

    if not hasattr(exp_config, "method"):
        raise ValueError("Experiment config must have a `method` field.")

    method_config = exp_config["method"]
    method = AutoMethod.from_config(
        method_name=method_config["method_name"]
        if method_config.get("method_name")
        else None,
        config_file_path=method_config["method_config_file_path"]
        if method_config.get("method_config_file_path")
        else None,
        **method_config.get("method_args", {}),
    )
    evaluator = Evaluator(exp_config["metrics"])

    eval_dict = run_experiment(dataset=dataset, method=method, evaluator=evaluator)
    print("Elapsed time:", time.time() - start_time)
    print(eval_dict)
    with open(os.path.join(save_dir, f"eval_{exp_name}.json"), "w") as f:
        json.dump(eval_dict, f)
