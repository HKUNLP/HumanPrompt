from typing import Any, List, Union


class Aggregate(object):
    @staticmethod
    def aggregate(
        answers: Union[str, List[str]], **kwargs: Any
    ) -> Union[str, List[str]]:
        """
        Aggregate the answers into a single answer.

        Args:
            answers: a list of answers
            **kwargs: other arguments

        Returns: a single answer

        """
        raise NotImplementedError
