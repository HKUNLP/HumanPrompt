from typing import Union, List, Dict, Callable
from .transform.transform_factory import TransformFactory


class PromptBuilder:
    """Prompt builder tool class."""

    @staticmethod
    def build_prompt(
            file_path: str = None,
            x: Union[str, Dict] = None,
            y: Union[str, Dict] = None,  # Union[List[str], List[Any[str, Dict]]]
            transform: Union[str, Callable] = None
    ):
        prompt = ""

        if file_path and not x:
            with open(file_path, 'r') as f:
                prompt += f.read()
            return prompt
        elif file_path and x:
            with open(file_path, 'r') as f:
                prompt += f.read()
                prompt += "\n\n"

        if isinstance(transform, Callable):
            if x and y:
                prompt += transform(x, y)
            elif x:
                prompt += transform(x)
            else:
                raise ValueError("x is required for transform")

            return prompt

        if isinstance(transform, str):
            if x and y:
                prompt += TransformFactory.get_transform(transform).transform(x, y)
            elif x:
                prompt += TransformFactory.get_transform(transform).transform(x)
            else:
                raise ValueError("x is required for transform")

            return prompt
