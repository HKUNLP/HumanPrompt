from typing import Any, Dict, List, Set

import evaluate


class Evaluator:
    """Evaluator base class."""

    _available_metrics: Set[str] = {"exact_match", "accuracy", "f1", "squad"}

    def __init__(
        self,
        metrics: List[str],
    ):
        self.metric_executors = {}
        for metric in metrics:
            if metric not in self._available_metrics:
                raise ValueError(f"Metric {metric} is not available.")
            # initialize metric executors
            self.metric_executors[metric] = evaluate.load(metric)

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

        for metric, executor in self.metric_executors.items():
            eval_dict.update(
                executor.compute(references=gold_answers, predictions=predictions)
            )

        return eval_dict
