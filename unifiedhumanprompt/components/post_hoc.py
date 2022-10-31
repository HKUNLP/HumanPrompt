from typing import Union, List, Dict, Callable
from .extract.extract_factory import ExtractFactory


class PostHoc(object):
    """Answer extract tool class"""

    @staticmethod
    def handle(
            raw_response: str = None,
            extract: Union[str, Callable] = None,
            **kwargs
    ):

        if isinstance(extract, Callable):
            return extract(raw_response, **kwargs)

        if isinstance(extract, str):
            return ExtractFactory.get_extract(extract).extract(raw_response, **kwargs)

        return raw_response
