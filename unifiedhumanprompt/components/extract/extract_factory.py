from .base import Extract
from .cot import CoTExtract


class ExtractFactory(object):
    """
    Provide built factory class for creating transforms.
    """
    current_extracts = {
        'default': Extract,
        'cot': CoTExtract,
        # Add more transforms here
    }

    @staticmethod
    def get_extract(extract: str) -> Extract:
        # If the extract is in current_extracts, return the identity class
        if extract in ExtractFactory.current_extracts:
            return ExtractFactory.current_extracts[extract]
        else:
            # If the extract is not in, treat it as a class name and return the class
            return globals()[extract]
