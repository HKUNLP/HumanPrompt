from typing import Any, Dict, Callable
from unifiedhumanprompt.components.transform.base import Transform
from transform.transform_factory import TransformFactory


class PromptBuilder(object):
    def __init__(self):
        pass

    @staticmethod
    def build_prompt(
            file_path: str = None,
            x: Any[str, Dict] = None,
            y: Any[str, Dict] = None, # Any[List[str], List[Any[str, Dict]]]
            transform: Any[str, Callable] = None
    ):
        if file_path:
            with open(file_path, 'r') as f:
                prompt = f.read()
            return prompt

        if isinstance(transform, Callable):
            if x and y:
                prompt = transform(x, y)
            elif x:
                prompt = transform(x)
            else:
                raise ValueError("x is required for transform")

            return prompt

        if isinstance(transform, str):
            if x and y:
                prompt = TransformFactory.get_transform(transform).transform(x, y)
            elif x:
                prompt = TransformFactory.get_transform(transform).transform(x)
            else:
                raise ValueError("x is required for transform")

            return prompt


x, y = {"question": 'my input'}, {"answer": 'my output'}
prompt = PromptBuilder.build_prompt(x=x, y=y, transform="cot")
