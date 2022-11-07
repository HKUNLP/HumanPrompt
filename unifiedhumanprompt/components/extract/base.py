from typing import Any


class Extract(object):
    @staticmethod
    def extract(raw_response: str, **kwargs: Any) -> str:
        """
        Extract the answer from the raw response.
        Args:
            raw_response: raw response from model
            **kwargs: other arguments
        Returns: extracted result
        """
        return raw_response.strip()
