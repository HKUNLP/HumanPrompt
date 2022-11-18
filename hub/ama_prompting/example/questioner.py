from typing import Any, Dict, Union

from humanprompt.components.transform.base import Transform


class QuestionerTransform(Transform):
    @staticmethod
    def transform(
        x: Union[str, Dict[str, str]],
        y: Union[str, Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """ """
        if isinstance(x, str) or isinstance(y, str):
            raise NotImplementedError

        transformed = f"Statement: {x['statement']}\n"
        transformed += "Question: "

        if y:
            transformed += f"{y['question']}"

        return transformed
