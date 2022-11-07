from typing import Any, List


class Aggregate(object):
    @staticmethod
    def aggregate(answers: List[str], **kwargs: Any) -> str:
        """
        Aggregate the answers into a single answer.

        Args:
            answers: a list of answers
            **kwargs: other arguments

        Returns: a single answer

        """
        raise NotImplementedError
