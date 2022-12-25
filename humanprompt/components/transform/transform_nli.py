from typing import Any, Dict, Union

from .transform_base import Transform


class NLITransform(Transform):
    @staticmethod
    def transform(
        x: Union[str, Dict], y: Union[str, Dict] = None, **kwargs: Any
    ) -> str:
        """
        Transform x and y into natural language inference(NLI) format.

        Args:
            x: a dict with keys "premise" and "hypothesis".
            y: a dict with keys "answer" and maybe others.
            **kwargs: other arguments

        Returns: a string of premise, hypothesis and answer.

        """
        assert isinstance(x, Dict)

        transformed = f"Premise: {x['premise']}\n"
        transformed += f"Hypothesis: {x['hypothesis']}\n"
        transformed += "Answer: "

        if y:
            assert isinstance(y, Dict)
            if "extraction_words" in kwargs:
                extraction_words = kwargs["extraction_words"]
            else:
                extraction_words = "The answer is"
            transformed += f"{extraction_words} {y['answer']}"

        return transformed
