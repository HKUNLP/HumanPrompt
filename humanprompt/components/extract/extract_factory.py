from typing import Type

from .extract_base import Extract
from .extract_regex import RegExtract

from hub.cot.commonsense_qa.extract_cot_commonsense_qa import CoTCommonsenseQAExtract


class ExtractFactory(object):
    """
    Provide built factory class for creating transforms.
    """

    current_extracts = {
        "default": Extract,
        "regex": RegExtract,
        # Method&Dataset-specific
        "cot-commonsense_qa": CoTCommonsenseQAExtract
    }

    @staticmethod
    def get_extract(extract: str) -> Type[Extract]:
        # If the extract is in current_extracts, return the identity class
        if extract in ExtractFactory.current_extracts:
            return ExtractFactory.current_extracts[extract]
        else:
            # If the extract is not in, treat it as a class name and return the class
            return globals()[extract]
