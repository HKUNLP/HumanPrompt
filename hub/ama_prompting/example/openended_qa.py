from typing import Any, Dict, Union

from humanprompt.components.transform.base import Transform


class OpenEndedQATransform(Transform):
    @staticmethod
    def transform(
        x: Union[str, Dict[str, str]],
        y: Union[str, Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """ """
        if isinstance(x, str) or isinstance(y, str):
            raise NotImplementedError

        transformed = f"Passage: {x['passage']}\nQuestion: {x['question']}"
        transformed += "Answer: "

        if y:
            transformed += f"{y['answer']}"

        return transformed
