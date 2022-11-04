from typing import Callable, Dict, Union

from .transform.transform_factory import TransformFactory


class PromptBuilder:
    """Prompt builder tool class."""

    @staticmethod
    def build_prompt(
        file_path: str = None,
        x: Union[str, Dict] = None,
        y: Union[str, Dict] = None,  # Union[List[str], List[Any[str, Dict]]]
        transform: Union[str, Callable] = None,
        **kwargs
    ):
        prompt = ""

        if file_path and not x:
            with open(file_path, "r") as f:
                prompt += f.read()
            return prompt
        elif file_path and x:
            with open(file_path, "r") as f:
                prompt += f.read()
                prompt += "\n\n"

        if isinstance(transform, Callable):
            if x and y:
                prompt += transform(x, y, **kwargs)
            elif x:
                prompt += transform(x, **kwargs)
            else:
                raise ValueError("x is required for transform")

            return prompt
        # TODO: use if/elif/else
        if isinstance(transform, str):
            if x and y:
                prompt += TransformFactory.get_transform(transform).transform(
                    x, y, **kwargs
                )
            elif x:
                prompt += TransformFactory.get_transform(transform).transform(
                    x, **kwargs
                )
            else:
                raise ValueError("x is required for transform")

            return prompt
