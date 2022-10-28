from typing import Callable
from transform.transform_factory import TransformFactory


class PromptBuilder():
    def __init__(self):
        pass

    @staticmethod
    def build_prompt(
            file_path: str = None,
            prompt: str = None,
            x: str = None,
            y: str = None,
            transform: Callable = None
    ):
        if file_path:
            with open(file_path, 'r') as f:
                prompt = f.read()
        elif x and y:
            prompt = transform(x, y)
        elif x:
            prompt = transform(x)
        elif prompt:
            pass


x, y = 'my input', 'my output'
prompt = PromptBuilder.build_prompt(x=x, y=y, transform=TransformFactory.get_transform(template='CoT'))

