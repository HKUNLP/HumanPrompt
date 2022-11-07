from typing import Type

from .base import Aggregate
from .simple_majority_vote import SimpleMajorityVote


class AggregateFactory(object):
    """
    Provide built factory class for creating transforms.
    """

    current_extracts = {
        "default": Aggregate,
        "simple_majority_vote": SimpleMajorityVote,
        # Add more transforms here
    }

    @staticmethod
    def get_aggregate(extract: str) -> Type[Aggregate]:
        # If the extract is in current_extracts, return the identity class
        if extract in AggregateFactory.current_extracts:
            return AggregateFactory.current_extracts[extract]
        else:
            # If the extract is not in, treat it as a class name and return the class
            return globals()[extract]
