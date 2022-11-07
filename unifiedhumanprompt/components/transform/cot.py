from typing import Any, Dict, Union

from .base import Transform


class CoTTransform(Transform):
    @staticmethod
    def transform(
        x: Union[str, Dict], y: Union[str, Dict] = None, **kwargs: Any
    ) -> str:
        """
        Chain of Thought (CoT) is a prompt format a series of intermediate reasoning steps
        which could significantly improves the ability of large language models to perform
        complex reasoning(https://arxiv.org/abs/2201.11903).

        Here is a basic implement of it.

        Args:
            x: input, could be a str or a dict, when it is a str, it is the question itself.
            y:  output, could be a str or a dict, when it is a str, it is the chain-of-thought and answer itself.
            **kwargs: other arguments

        Returns: a string of prompt

        """
        transformed = f"Q: {x['question']}\n"
        if "context" in x:
            transformed += f"{x['context']}\n"
        transformed += "A: "

        if y:
            if "extraction_words" in kwargs:
                extraction_words = kwargs["extraction_words"]
            else:
                extraction_words = "The answer is"
            transformed += f"{y['chain_of_thought']} {extraction_words} {y['answer']}"

        return transformed
