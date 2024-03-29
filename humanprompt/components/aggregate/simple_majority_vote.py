from typing import Any, List, Union

from .base import Aggregate


class SimpleMajorityVote(Aggregate):
    @staticmethod
    def aggregate(
        answers: Union[str, List[str]], **kwargs: Any
    ) -> Union[str, List[str]]:
        """
        Aggregate the answers into a single answer by majority voting.

        Args:
            answers: a list of answers
            **kwargs: other arguments

        Returns: a single answer

        """
        # Get the count each type of answers
        answer_count = {}
        for answer in answers:
            if isinstance(answer, str):
                if (
                    answer == "<error>"
                    or answer == ""
                    or answer == "[]"
                    or answer == set()
                ):
                    continue
                if answer not in answer_count:
                    answer_count[answer] = 0
                answer_count[answer] += 1
            elif isinstance(answer, List):
                if answer == ["<error>"] or answer == []:
                    continue
                if tuple(answer) not in answer_count:
                    answer_count[tuple(answer)] = 0
                answer_count[tuple(answer)] += 1
        try:
            vote_result = max(answer_count.items(), key=lambda x: x[1])[0]
        except ValueError:
            vote_result = "0"

        if isinstance(vote_result, str):
            return vote_result
        elif isinstance(vote_result, tuple):
            return list(vote_result)
        else:
            assert False, "Invalid vote result type: {}".format(type(vote_result))
