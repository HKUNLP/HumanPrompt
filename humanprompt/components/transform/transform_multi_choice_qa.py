from typing import Any, Dict, Union

from .transform_base import Transform


class MultiChoiceQATransform(Transform):
    @staticmethod
    def transform(
        x: Union[str, Dict], y: Union[str, Dict] = None, **kwargs: Any
    ) -> str:
        """
        Transform x and y into a multi-choice format.

        Args:
            x: a dict with keys "question" and "choices", where "choices" is a list of labels and text.
            y: a dict with keys "answer" and maybe others.
            **kwargs: other arguments

        Returns: a string of question, choices, and answer

        """
        assert isinstance(x, Dict)

        transformed = f"Q: {x['question']}\n"
        transformed += "Answer choices: {}\n".format(
            " ".join(
                [
                    "({}) {}".format(label.lower(), text.lower())
                    for label, text in zip(x["choices"]["label"], x["choices"]["text"])
                ]
            )
        )
        transformed += "A: "

        if y:
            assert isinstance(y, Dict)
            if "extraction_words" in kwargs:
                extraction_words = kwargs["extraction_words"]
            else:
                extraction_words = "The answer is"
            transformed += f"{extraction_words} {y['answer']}"

        return transformed
