from typing import Any, List, Dict

import evaluate


class Evaluator:
    """Evaluator base class."""

    _available_metrics: set[str] = {
            "exact_match",
            "accuracy",
            "f1"
         }

    def __init__(
            self,
            metrics: List[str],
    ):
        for metric in metrics:
            if metric not in self._available_metrics:
                raise ValueError(f"Metric {metric} is not available.")

        self.metrics = metrics

    def evaluate(
            self,
            predictions: List,  # TODO: Extend input to `Union[List[Dict], Dict[str, Dict]`
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
        # TODO: Add support for other metrics, maybe not in `evaluate` package.
        eval_dict = dict()
        if "exact_match" in self.metrics:
            metric_exact_match = evaluate.load("exact_match")
            eval_dict.update(metric_exact_match.compute(references=gold_answers, predictions=predictions))
        if "accuracy" in self.metrics:
            metric_accuracy = evaluate.load("accuracy")
            eval_dict.update(metric_accuracy.compute(references=gold_answers, predictions=predictions))
        if "f1" in self.metrics:
            metric_f1 = evaluate.load("f1")
            eval_dict.update(metric_f1.compute(references=gold_answers, predictions=predictions))

        return eval_dict
