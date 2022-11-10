from typing import Any, List

from .base import Aggregate


class SimpleMajorityVote(Aggregate):
    @staticmethod
    def aggregate(answers: List[str], **kwargs: Any) -> str:
        """
        Aggregate the answers into a single answer.

        Args:
            answers: a list of answers
            **kwargs: other arguments

        Returns: a single answer

        """
        return max(set(answers), key=answers.count)
