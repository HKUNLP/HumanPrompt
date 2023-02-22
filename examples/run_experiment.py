import argparse
import json
import os
import time
from typing import Dict

import openai
from datasets import Dataset

from humanprompt.evaluators.evaluator import Evaluator
from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.methods.base_method.method import PromptMethod
from humanprompt.tasks.dataset_loader import DatasetLoader
from humanprompt.utils.config_utils import load_config


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
        data_item["idx"] = idx
        if data_item.get("id", None) is None:
            data_item["id"] = idx
        if use_cache and os.path.exists(
            os.path.join(tmp_save_dir, f"{idx}_{data_item['id']}.json")
        ):
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
                    current_key = os.environ["OPENAI_API_KEY"]
                    print("Using OpenAI key: ", current_key)
                    prediction = method.run(x=data_item, verbose=verbose)
                    break
                except openai.error.OpenAIError as e:
                    print(f"Error when getting response: {e}")
                    continue
            if prediction is None:
                prediction = "<empty>"
            # Answer post-processing for evaluation
            prediction = evaluator.normalize_answer(prediction)
            gold_answer = evaluator.normalize_answer(data_item["answer"])
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
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", type=str, default="cot-gsm8k", help="Experiment name."
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=None,
        help="Number of test samples. Set None to use all.",
    )
    parser.add_argument(
        "--debug_indices",
        type=str,
        default=None,
        help="Debug indices of samples in dataset. Set None to use all.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--use_cache",
        type=bool,
        default=True,
        help="Whether to use cache for already tested samples.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to print verbose information."
    )
    args = parser.parse_args()

    # Meta-config
    exp_name = args.exp_name
    exp_config = load_config(f"configs/{exp_name}.yaml")
    num_test_samples = args.num_test_samples
    debug_indices = (
        args.debug_indices
        if args.debug_indices is None
        else [int(x) for x in args.debug_indices.split(",")]
    )
    save_dir = args.save_dir
    tmp_save_dir = os.path.join(save_dir, "tmp", f"{exp_name}/")
    use_cache = args.use_cache
    verbose = args.verbose

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
    if num_test_samples:
        dataset = dataset.select(range(num_test_samples))
    if debug_indices:
        dataset = dataset.select(debug_indices)

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
    evaluator = Evaluator(
        metrics=exp_config["metrics"],
        dataset_name=dataset_config["dataset_name"],
        dataset_subset_name=dataset_config["dataset_subset_name"]
        if "dataset_subset_name" in dataset_config
        else None,
    )

    start_time = time.time()
    eval_dict = run_experiment(dataset=dataset, method=method, evaluator=evaluator)
    print("Elapsed time:", time.time() - start_time)
    print(eval_dict)
    with open(os.path.join(save_dir, f"eval_{exp_name}.json"), "w") as f:
        json.dump(eval_dict, f)
