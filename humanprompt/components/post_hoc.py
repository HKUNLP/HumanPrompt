import warnings
from typing import Any, Callable, List, Union

from .aggregate.aggregate_factory import AggregateFactory
from .extract.extract_factory import ExtractFactory


class HocPoster(object):
    """Answer extract&aggregation tool class"""

    @staticmethod
    def post_hoc(
        raw_response: Union[str, List] = None,
        extract: Union[str, Callable] = None,
        aggregation: Union[str, Callable] = None,
        post_hoc: Union[str, Callable] = None,
        **kwargs: Any
    ) -> Union[str, List[Any]]:
        assert bool(extract or aggregation) ^ bool(
            post_hoc
        ), "When assigning post_hoc, extract and aggregation should not be assigned."

        def _post_hoc_for_str(
            _post_hoc: Union[str, Callable], _raw_response: str
        ) -> str:
            if _post_hoc:
                if isinstance(_post_hoc, str):
                    raise NotImplementedError("post_hoc factory is not implemented.")
                elif callable(_post_hoc):
                    return _post_hoc(_raw_response, **kwargs)
                else:
                    raise TypeError("post_hoc should be str or Callable.")
            else:
                if isinstance(extract, str):
                    y = ExtractFactory.get_extract(extract).extract(
                        _raw_response, **kwargs
                    )
                elif callable(_post_hoc):
                    y = extract(_raw_response, **kwargs)
                else:
                    # It is assumed that the raw response is the answer with no need to extract.
                    y = _raw_response

                return y

        if isinstance(raw_response, str):
            if aggregation:
                warnings.warn(
                    "Aggregation is not supported for str type raw_response. Skip it this time.",
                    UserWarning,
                )
            return _post_hoc_for_str(post_hoc, raw_response)

        if isinstance(raw_response, List):
            try:
                if isinstance(post_hoc, str):
                    raise NotImplementedError("post_hoc factory is not implemented.")
                elif callable(post_hoc):
                    # First try to use the post_hoc function passed in to handle the raw_responses
                    return post_hoc(raw_response, **kwargs)
                else:
                    raise TypeError("post_hoc should be str or Callable.")
            except Exception:
                # If the post_hoc function is not defined, use the extract and aggregation functions to
                # handle the raw_responses
                extracted_y_s = [
                    _post_hoc_for_str(post_hoc, raw_response_str)
                    for raw_response_str in raw_response
                ]
                if aggregation:
                    if isinstance(aggregation, str):
                        return AggregateFactory.get_aggregate(aggregation).aggregate(
                            extracted_y_s, **kwargs
                        )
                    elif callable(aggregation):
                        return aggregation(extracted_y_s, **kwargs)
                else:
                    return extracted_y_s

        raise ValueError("Nothing done for post_hoc.")
