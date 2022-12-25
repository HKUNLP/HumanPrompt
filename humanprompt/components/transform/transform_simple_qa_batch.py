from typing import Any, Dict, Union, List

from .transform_base import Transform


class QABatchTransform(Transform):
    @staticmethod
    def transform(
            x: List[Dict], y: List[Dict] = None, **kwargs: Any
    ) -> str:
        """
        Transform x and y into a simple QA format.

        Args:
            x: a dict with keys "question" and maybe others.
            y: a dict with keys "answer" and maybe others.
            **kwargs: other arguments

        Returns: a string of question, choices, and answer

        """
        if not isinstance(x[0], Dict) \
                or (y and not isinstance(y[0], Dict)):
            raise TypeError("x and y should be dict in QA (batch inference) task.")

        transformed = ""
        for idx, x_ in enumerate(x, 1):
            transformed += f"Q[{idx}]: {x_['question']}\n"
        transformed += "A[1]: "

        if y:
            # TODO
            pass

        return transformed
