from typing import Union, List, Dict, Callable
from .extract.extract_factory import ExtractFactory
from .aggregate.aggregate_factory import AggregateFactory


class HocPoster(object):
    """Answer extract&aggregation tool class"""

    @staticmethod
    def post_hoc(
            raw_response: Union[str, List] = None,
            extract: Union[str, Callable] = None,
            aggregation: Union[str, Callable] = None,
            post_hoc: Union[str, Callable] = None,
            **kwargs
    ):
        assert bool(extract or aggregation) \
               ^ bool(post_hoc), "When assigning post_hoc, extract and aggregation should not be assigned."

        def _post_hoc_for_str(_raw_response):
            if post_hoc:
                if isinstance(post_hoc, Callable):
                    return post_hoc(_raw_response, **kwargs)
                elif isinstance(post_hoc, str):
                    raise NotImplementedError("post_hoc factory is not implemented.")
            else:
                if isinstance(extract, Callable):
                    y = extract(_raw_response, **kwargs)
                elif isinstance(extract, str):
                    y = ExtractFactory.get_extract(extract).extract(_raw_response, **kwargs)
                else:
                    # It is assumed that the raw response is the answer with no need to extract.
                    y = _raw_response

                return y

        if isinstance(raw_response, str):
            if aggregation:
                import warnings
                warnings.warn("Aggregation is not supported for str type raw_response. Skip it this time.", UserWarning)
            return _post_hoc_for_str(raw_response)

        if isinstance(raw_response, List):
            try:
                # First try to use the post_hoc function passed in to handle the raw_responses
                return post_hoc(raw_response, **kwargs)
            except:
                # If the post_hoc function is not defined, use the extract and aggregation functions to
                # handle the raw_responses
                extracted_y_s = [_post_hoc_for_str(raw_response_str) for raw_response_str in raw_response]
                if aggregation:
                    if isinstance(aggregation, Callable):
                        return aggregation(extracted_y_s, **kwargs)
                    elif isinstance(aggregation, str):
                        return AggregateFactory.get_aggregate(aggregation).aggregate(extracted_y_s, **kwargs)
                else:
                    return extracted_y_s

