from .base import Aggregate


class SimpleMajorityVote(Aggregate):

    @staticmethod
    def aggregate(answers, **kwargs):
        # select the item of answers with the most frequency
        return max(set(answers), key=answers.count)
