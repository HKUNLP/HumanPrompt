from typing import Any, Dict, List, Optional, Union

from ...components.prompt import PromptBuilder
from ..base_method.method import PromptMethod


class DBText2SQLMethod(PromptMethod):
    """TODO: add docstring"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def run(
        self,
        x: Union[str, Dict],
        in_context_examples: List[Dict] = None,
        prompt_file_path: Optional[str] = None,
        **kwargs: Any
    ) -> Union[str, List[str]]:
        prompt = PromptBuilder.build_prompt(
            x=x,
            transform=kwargs["transform"]
            if "transform" in kwargs
            else self.kwargs.get("transform", None),
            prompt_style=kwargs["prompt_style"]
            if "prompt_style" in kwargs
            else self.kwargs.get("prompt_style", None),
        )

        y = self.run_lm(prompt, **kwargs)

        return y
