from typing import Type

from .base import Extract
from .regex import RegExtract


class ExtractFactory(object):
    """
    Provide built factory class for creating transforms.
    """

    current_extracts = {
        "default": Extract,
        "regex": RegExtract,
        # Add more transforms here
    }

    @staticmethod
    def get_extract(extract: str) -> Type[Extract]:
        # If the extract is in current_extracts, return the identity class
        if extract in ExtractFactory.current_extracts:
            return ExtractFactory.current_extracts[extract]
        else:
            # If the extract is not in, treat it as a class name and return the class
            return globals()[extract]
