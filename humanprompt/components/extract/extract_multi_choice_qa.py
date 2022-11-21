import re
from typing import Any

from .extract_base import Extract


class MultiChoiceQAExtract(Extract):
    @staticmethod
    def extract(raw_response: str, **kwargs: Any) -> str:
        """
        Extract the multi-choice answer from raw_response by regex.

        Args:
            raw_response: raw response from model
            **kwargs: other arguments
        Returns: extracted result

        """
        answer = raw_response.strip()
        if "extraction_regex" in kwargs:
            # if extraction_words is specified, we use it to extract the answer
            extraction_regex = kwargs["extraction_regex"]
            answer = re.match(extraction_regex, raw_response).group(1)

        answer = answer.lstrip('(').rstrip(')').lower()

        return answer.strip()
