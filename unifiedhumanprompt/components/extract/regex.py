import re
from typing import Any

from .base import Extract


class RegExtract(Extract):
    @staticmethod
    def extract(raw_response: str, **kwargs: Any) -> str:
        """
        Extract the answer from raw_response by regex.

        Chain of Thought (CoT) is a prompt format a series of intermediate reasoning steps
        which could significantly improves the ability of large language models to perform
        complex reasoning(https://arxiv.org/abs/2201.11903).

        The response contains a chain of thought and an answer, so we need to extract the answer from the response.

        Args:
            raw_response: raw response from model
            **kwargs: other arguments
        Returns: extracted result

        """

        if "extraction_regex" in kwargs:
            # if extraction_words is specified, we use it to extract the answer
            extraction_regex = kwargs["extraction_regex"]
            answer = re.match(extraction_regex, raw_response).group(1)
            return answer.strip()

        return raw_response.strip()
