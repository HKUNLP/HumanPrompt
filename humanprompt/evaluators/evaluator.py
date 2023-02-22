from typing import Any, Dict, List, Set, Union
import re

import evaluate


class Evaluator:
    """Evaluator base class."""

    _available_metrics: Set[str] = {"exact_match", "accuracy", "f1", "squad"}

    def __init__(
        self,
        metrics: List[str],
        dataset_name: str = None,
        dataset_subset_name: str = None
    ):
        self.metric_executors = {}
        for metric in metrics:
            if metric not in self._available_metrics:
                raise ValueError(f"Metric {metric} is not available.")
            # initialize metric executors
            self.metric_executors[metric] = evaluate.load(metric)
        self.dataset_name = dataset_name
        self.dataset_subset_name = dataset_subset_name

    def evaluate(
        self,
        predictions: List,
        gold_answers: List,
    ) -> Dict[str, Any]:
        """
        Evaluate the predictions with assigned metrics.

        Args:
            predictions: List of predictions.
            gold_answers: List of gold answers.
        Returns:
            A dictionary of evaluation metrics.
        """
        # TODO: Add support for other metrics not in `evaluate` package.
        eval_dict = dict()

        for metric, executor in self.metric_executors.items():
            eval_dict.update(
                executor.compute(references=gold_answers, predictions=predictions)
            )

        return eval_dict

    def normalize_answer(
            self,
            answer: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        """
        Normalize answers according to the dataset.
        Reference https://github.com/kojima-takeshi188/zero_shot_cot/blob/main/utils.py

        Args:
            answer: Answer to be normalized.
        Returns:
            Normalized answer.
        """
        if not isinstance(answer, list):
            answer = [answer]
        answer = [str(a).lower() for a in answer]

        for idx, a in enumerate(answer):
            try:
                if self.dataset_name in ("gsm8k", "add_sub", "multi_arith", "svamp"):
                    a = a.replace(",", "")
                    a = re.findall(r'-?\d+\.?\d*', a)[-1]
                    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
                    if a != "":
                        if a[-1] == ".":
                            a = a[:-1]
                    if a[-2:] == ".0":
                        a = a[:-2]
                elif self.dataset_name in ("strategy_qa"):
                    if a == "yes":
                        a = "true"
                    elif a == "no":
                        a = "false"
                elif self.dataset_name == "glue" and self.dataset_subset_name == "rte":
                    if a == "true":
                        a = "0"
                    elif a == "false":
                        a = "1"
                elif self.dataset_name == "glue" and self.dataset_subset_name == "mnli":
                    if a == "true":
                        a = "0"
                    elif a == "neutral":
                        a = "1"
                    elif a == "false":
                        a = "2"
                elif self.dataset_name == "SetFit/sst5":
                    if a == "very negative":
                        a = "0"
                    elif a == "negative":
                        a = "1"
                    elif a == "neutral":
                        a = "2"
                    elif a == "positive":
                        a = "3"
                    elif a == "very positive":
                        a = "4"
            except Exception as e:
                pass
            answer[idx] = a

        if len(answer) == 1:
            return answer[0]
        else:
            return answer
