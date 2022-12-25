from typing import Any, Dict, Union

from .transform_base import Transform


class QATransform(Transform):
    @staticmethod
    def transform(
        x: Union[str, Dict], y: Union[str, Dict] = None, **kwargs: Any
    ) -> str:
        """
        Transform x and y into a question and answer format.

        Here is a basic implement of it.

        Args:
            x: question, could be a str or a dict, when it is a str, it is the question itself.
            y: answer, could be a str or a dict, when it is a str, it is the answer itself.
            **kwargs: other arguments

        Returns: a string of question and answer

        """
        if y is None:
            if isinstance(x, str):
                return f"Q: {x}\nA: "
            elif isinstance(x, dict):
                return f"Q: {x['question']}\nA: "
            else:
                raise ValueError("x is not a str or a dict")
        else:
            if isinstance(x, str) and isinstance(y, str):
                return f"Q: {x}\nA: {y}"
            elif isinstance(x, dict) and isinstance(y, dict):
                return f"Q: {x['question']}\nA: {y['answer']}"
            else:
                raise ValueError("x and y are not both str or both dict")
