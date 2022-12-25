from typing import Any, Dict, Union, List

from .transform_base import Transform


class NLIBatchTransform(Transform):
    @staticmethod
    def transform(
            x: List[Dict], y: List[Dict] = None, **kwargs: Any
    ) -> str:
        """
        Transform x and y into natural language inference(NLI) format.

        Args:
            x: a dict with keys "premise", "hypothesis" and maybe others.
            y: a dict with keys "answer" and maybe others.
            **kwargs: other arguments

        Returns: a string of premise, hypothesis and answer.

        """
        if not isinstance(x[0], Dict) \
                or (y and not isinstance(y[0], Dict)):
            raise TypeError("x and y should be dict in NLI (batch inference) task.")

        transformed = ""
        for idx, x_ in enumerate(x, 1):
            transformed += f"Premise[{idx}]: {x_['premise']}\n"
            transformed += f"Hypothesis[{idx}]: {x_['hypothesis']}\n"
        transformed += "Answer[1]: "

        if y:
            # TODO
            pass

        return transformed
